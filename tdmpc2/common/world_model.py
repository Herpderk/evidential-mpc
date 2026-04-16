from copy import deepcopy
from typing import NamedTuple
from math import pi as PI

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class NormalInverseGamma(NamedTuple):
    mu: torch.Tensor
    lam: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		self._target_flow = layers.acf(
			feature_dim=cfg.latent_dim,
			context_dim=cfg.latent_dim + cfg.action_dim + cfg.task_dim,
			conditioner_dims=max(cfg.num_targetflow_cond_layers, 1) * [cfg.targetflow_cond_dim],
			num_layers=cfg.num_targetflow_layers,
		)
		self._evidence_flow = layers.acf(
			feature_dim=cfg.latent_dim + cfg.action_dim,
			context_dim=0,
			conditioner_dims=max(cfg.num_evidenceflow_cond_layers, 1) * [cfg.evidenceflow_cond_dim],
			num_layers=cfg.num_evidenceflow_layers,
		)
		self._dynamics = layers.mlp(
            in_dim=cfg.latent_dim + cfg.action_dim + cfg.task_dim,
            mlp_dims=2*[cfg.mlp_dim],
            out_dim=2*cfg.latent_dim,    # Output posterior update params
            #act=layers.SimNorm(cfg),
        )
		self._simnorm = layers.SimNorm(cfg)

		self.register_buffer(
      		"_certainty_budget",
			(cfg.latent_dim+cfg.action_dim) * torch.log(torch.tensor(4*PI)))
		self.register_buffer("_evidence_prior", torch.tensor(1.0))  # Prior evidence for conjugate update in dynamics
		self.register_buffer("_sufstat_prior", torch.cat([
	  		torch.zeros(cfg.latent_dim),
		 	100.0 * torch.ones(cfg.latent_dim)], dim=-1))  # Prior parameters for conjugate update in dynamics

		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

		# Need to initialize the final conditioner layers to 0
		for flow_layer in self._target_flow.flows:
			if hasattr(flow_layer, 'zero_final_conditioner_layer'):
				flow_layer.zero_final_conditioner_layer()
		for flow_layer in self._evidence_flow.flows:
			if hasattr(flow_layer, 'zero_final_conditioner_layer'):
				flow_layer.zero_final_conditioner_layer()

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		# We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Target flow', 'Evidence flow', 'Noise-space dynamics',  'Reward', 'Termination', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._target_flow, self._evidence_flow, self._dynamics, self._reward, self._termination, self._pi, self._Qs]):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def latent2noise(self, z_next, z_prev, a, task):
		if self.cfg.multitask:
			z_prev = self.task_emb(z_prev, task)
		z_prev = torch.cat([z_prev, a], dim=-1)
		return self._target_flow.inverse(z_next, context=z_prev)

	def noise2latent(self, u_next, z_prev, a, task):
		if self.cfg.multitask:
			z_prev = self.task_emb(z_prev, task)
		z_prev = torch.cat([z_prev, a], dim=-1)
		return self._target_flow.forward(u_next, context=z_prev)

	def target_noise_and_loss(self, z_next, z_prev, a, task):
		if self.cfg.multitask:
			z_prev = self.task_emb(z_prev, task)
		z_prev = torch.cat([z_prev, a], dim=-1)

		log_q = torch.zeros(len(z_next), device=z_next.device)
		u_next = z_next
		for i in range(len(self._target_flow.flows) - 1, -1, -1):
			u_next, log_det = self._target_flow.flows[i].inverse(u_next, context=z_prev)
			log_q += log_det
		log_q += self._target_flow.q0.log_prob(u_next, context=z_prev)
		return u_next, -torch.mean(log_q)

	def nll_under_conjugate_prior(self, y, evid_pred: NormalInverseGamma):
		mu, lam, alpha, beta = evid_pred
		squared_error = (mu - y) ** 2
		nll_under_cj = -0.5 * (
			-squared_error*alpha/beta - lam.reciprocal()
			+ torch.digamma(alpha) - torch.log(beta) - torch.log(torch.tensor(2*PI, device=y.device))
		)
		return nll_under_cj.mean()

	def conjugate_prior_entropy(self, evid_pred: NormalInverseGamma):
		"""
		Computes the entropy of the conjugate prior distribution in closed form.
		"""
		mu, lam, alpha, beta = evid_pred
		cj_entropy = torch.zeros(lam.shape[0], device=lam.device)
		LOG_2PI = torch.log(torch.tensor(2 * PI, device=lam.device))
		entropy_large = (
			1.0 + LOG_2PI - 2.0 * torch.log(alpha) + 1.5 * torch.log(beta) - 0.5 * torch.log(lam)
		)
		entropy_small = (
			0.5 - 0.5 * torch.log(lam) + alpha - (alpha + 1.5) * torch.digamma(alpha)
			+ 0.5 * LOG_2PI + 1.5 * torch.log(beta) + torch.lgamma(alpha)
		)
		cj_entropy = torch.where(alpha > 1e4, entropy_large, entropy_small)
		return cj_entropy.mean()

	def next_evidential_noise(self, z, a, task) -> NormalInverseGamma:
		"""
		Predicts the next state and its uncertainties in conditional noise space.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		sufstat_mean, sufstat_var = self._dynamics(z).chunk(2, dim=-1)	# Split sufficient statistics into mean and variance parts
		sufstat_mean = self._simnorm(sufstat_mean)	# Apply simnorm to mean part only
		sufstat_var = F.softplus(sufstat_var) + 1e-6 # Force variance to be positive
		sufstat_update = torch.cat([sufstat_mean, sufstat_var], dim=-1)

		log_evid_update = torch.log(self._certainty_budget) + self._evidence_flow.log_prob(z)[:,None]
		log_evid_combined = torch.cat([torch.log(self._evidence_prior).expand_as(log_evid_update), log_evid_update], dim=-1)
		evid_post = torch.logsumexp(log_evid_combined, dim=-1, keepdim=True).exp()

		sufstat_post_weights = torch.softmax(log_evid_combined, dim=-1)
		w_prior = sufstat_post_weights[:, 0:1]
		w_update = sufstat_post_weights[:, 1:2]
		sufstat_post = w_prior * self._sufstat_prior + w_update * sufstat_update

		# Derive conjugate prior distribution from posterior sufstateters
		sufstat_post_1, sufstat_post_2 = sufstat_post.chunk(2, dim=-1)
		mu = sufstat_post_1
		lam = evid_post
		alpha = evid_post / 2
		beta = 0.5 * evid_post * (sufstat_post_2 - sufstat_post_1**2)
		return NormalInverseGamma(mu, lam, alpha, beta)

	def next_noise(self, z, a, task):
		"""
		Predicts the next state in conditional noise space.
		"""
		evidential_pred = self.next_evidential_noise(z, a, task)
		return evidential_pred.mu

	def next_latent(self, z, a, task):
		"""
		Predicts the next state in latent space.
		"""
		u_next = self.next_noise(z, a, task)
		return self.noise2latent(u_next, z, a, task)

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)

	def termination(self, z, task, unnormalized=False):
		"""
		Predicts termination signal.
		"""
		assert task is None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask: # Mask out unused action dimensions
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2
