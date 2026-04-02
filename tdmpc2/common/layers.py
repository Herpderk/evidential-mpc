import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules
from copy import deepcopy

import normflows as nf


class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		# combine_state_for_ensemble causes graph breaks
		self.params = from_modules(*modules, as_module=True)
		with self.params[0].data.to("meta").to_module(modules[0]):
			self.module = deepcopy(modules[0])
		self._repr = str(modules[0])
		self._n = len(modules)

	def __len__(self):
		return self._n

	def _call(self, params, *args, **kwargs):
		with params.to_module(self.module):
			return self.module(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

	def __repr__(self):
		return f'Vectorized {len(self)}x ' + self._repr


class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"



class ConditionalCouplingConditioner(nn.Module):
	def __init__(self, feature_dim, context_dim, flow_dim, act=None):
		super().__init__()
		in_dim = (feature_dim // 2) + context_dim
		out_dim = feature_dim
		if act is None:
			act = nn.SiLU()
		self.mlp = nn.Sequential(
			NormedLinear(in_dim, flow_dim, act=act),
			NormedLinear(flow_dim, flow_dim, act=act),
			nn.Linear(flow_dim, out_dim) # MUST REMAIN PURE LINEAR
		)
		self.zero_final_layer()

	def zero_final_layer(self):
		# Crucial for normalizing flows: initialize last layer with zeros
		# This ensures the flow starts as an identity transformation
		nn.init.zeros_(self.mlp[-1].weight)
		nn.init.zeros_(self.mlp[-1].bias)

	def forward(self, half_feature, context=None):
		return self.mlp(torch.cat([half_feature, context], dim=-1))


class ConditionalAffineCoupling(nf.flows.Flow):
    def __init__(self, feature_dim, context_dim, flow_dim, use_preproc_logit=False, logit_eps=1e-6, act=None):
        super().__init__()
        self.use_preproc_logit = use_preproc_logit
        self.logit_eps = abs(logit_eps)
        self.conditioner = ConditionalCouplingConditioner(feature_dim, context_dim, flow_dim, act=act)

    def forward(self, feature, context=None):
        if self.use_preproc_logit:	# Pre-processing step for unbounding features between 0 and 1
            # Contract feature away from 0 and 1 and apply logit transform
            y = self.logit_eps + (1 - 2 * self.logit_eps) * feature
            feature = torch.log(y / (1 - y))

        # Split feature in half
        feat_1, feat_2 = feature.chunk(2, dim=-1)

        # Get scale and shift
        scale, shift = self.conditioner(feat_1, context=context).chunk(2, dim=-1)
        scale = torch.tanh(scale)

        # Transform feat_2
        feat_2 = feat_2 * torch.exp(scale) + shift
        feat_out = torch.cat([feat_1, feat_2], dim=-1)

        # Return transformed feature and log determinant (volume change)
        log_det = torch.sum(scale, dim=-1)
        return feat_out, log_det

    def inverse(self, feature, context=None):
        # Split feature in half
        feat_1, feat_2 = feature.chunk(2, dim=-1)

        # Get scale and shift (MUST use feat_1, which is unchanged)
        scale, shift = self.conditioner(feat_1, context=context).chunk(2, dim=-1)
        scale = torch.tanh(scale)

        # Inverse transform feat_2
        feat_2 = (feat_2 - shift) * torch.exp(-scale)
        feat_out = torch.cat([feat_1, feat_2], dim=-1)

        # Inverse log determinant of the coupling layer
        log_det = -torch.sum(scale, dim=-1)

		# We don't apply the inverse of the logit transform since we treat it like a pre-processing step
        return feat_out, log_det


class ConditionalReverse(nf.flows.Flow):
    def forward(self, z, context=None):
        # Zero log-det because swapping doesn't change volume
        log_det = torch.zeros(*z.shape[:-1], device=z.device)
        return z.flip(dims=[-1]), log_det

    def inverse(self, z, context=None):
        log_det = torch.zeros(*z.shape[:-1], device=z.device)
        return z.flip(dims=[-1]), log_det


def acf(feature_dim, context_dim, flow_dim, num_layers, act=None):
	flows = []
	for i in range(num_layers):
		if i == 0:
			use_preproc_logit = True
		else:
			use_preproc_logit = False
		flows += [ConditionalAffineCoupling(feature_dim, context_dim, flow_dim, use_preproc_logit=use_preproc_logit, act=act)]
		flows += [ConditionalReverse()]
	q0 = nf.distributions.DiagGaussian(feature_dim, trainable=False)
	return nf.ConditionalNormalizingFlow(q0=q0, flows=flows)


def maf(feature_dim, context_dim, flow_dim, num_layers, num_blocks=2):
    flows = []
    for i in range(num_layers):
        flows += [nf.flows.MaskedAffineAutoregressive(feature_dim, flow_dim,
                                                    context_features=context_dim,
                                                    num_blocks=num_blocks)]
        flows += [nf.flows.LULinearPermute(feature_dim)]
    q0 = nf.distributions.DiagGaussian(feature_dim, trainable=False)
    return nf.ConditionalNormalizingFlow(q0, flows)


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
	layers = [
		ShiftAug(), PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
		elif k == 'rgb':
			out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
	"""
	Converts a checkpoint from our old API to the new torch.compile compatible API.
	"""
	# check whether checkpoint is already in the new format
	if "_detach_Qs_params.0.weight" in source_state_dict:
		return source_state_dict

	name_map = ['weight', 'bias', 'ln.weight', 'ln.bias']
	new_state_dict = dict()

	# rename keys
	for key, val in list(source_state_dict.items()):
		if key.startswith('_Qs.'):
			num = key[len('_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_Qs.params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val
			new_total_key = "_detach_Qs_params." + new_key
			new_state_dict[new_total_key] = val
		elif key.startswith('_target_Qs.'):
			num = key[len('_target_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_target_Qs_params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val

	# add batch_size and device from target_state_dict to new_state_dict
	for prefix in ('_Qs.', '_detach_Qs_', '_target_Qs_'):
		for key in ('__batch_size', '__device'):
			new_key = prefix + 'params.' + key
			new_state_dict[new_key] = target_state_dict[new_key]

	# check that every key in new_state_dict is in target_state_dict
	for key in new_state_dict.keys():
		assert key in target_state_dict, f"key {key} not in target_state_dict"
	# check that all Qs keys in target_state_dict are in new_state_dict
	for key in target_state_dict.keys():
		if 'Qs' in key:
			assert key in new_state_dict, f"key {key} not in new_state_dict"
	# check that source_state_dict contains no Qs keys
	for key in source_state_dict.keys():
		assert 'Qs' not in key, f"key {key} contains 'Qs'"

	# copy log_std_min and log_std_max from target_state_dict to new_state_dict
	new_state_dict['log_std_min'] = target_state_dict['log_std_min']
	new_state_dict['log_std_dif'] = target_state_dict['log_std_dif']
	if '_action_masks' in target_state_dict:
		new_state_dict['_action_masks'] = target_state_dict['_action_masks']

	# copy new_state_dict to source_state_dict
	source_state_dict.update(new_state_dict)

	return source_state_dict
