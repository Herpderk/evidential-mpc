import torch
import numpy as np
import hydra
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from common.evidential import aleatoric_uncertainty, epistemic_uncertainty
from common.parser import parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2

# Observation indices
HEIGHT_IDX = 1
ANGLE_IDX = 2

# Line parameters (Only varying angle now)
CONSTANT_HEIGHT = 0.5  # Adjust this to your cheetah's typical resting height
ANGLE_RANGE = (-np.pi, np.pi)
RESOLUTION = 50

@hydra.main(config_name='config', config_path='.')
def evaluate_world_model_line(cfg: dict):
    """Loads the TD-MPC2 agent and evaluates uncertainty across pitch angles."""
    cfg = parse_cfg(cfg)

    env = make_env(cfg)
    agent = TDMPC2(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
    agent.load(cfg.checkpoint)

    print(f"Generating pixel-based state line at constant height {CONSTANT_HEIGHT}...")

    # 1D array of angles
    angles = torch.linspace(ANGLE_RANGE[0], ANGLE_RANGE[1], RESOLUTION)
    num_states = len(angles)

    obs_list = []
    env.reset()

    # Grab the underlying dm_control physics engine
    physics = None
    curr_env = env

    while curr_env is not None:
        if hasattr(curr_env, 'physics'):
            physics = curr_env.physics
            break
        elif hasattr(curr_env, '_env') and hasattr(curr_env._env, 'physics'):
            physics = curr_env._env.physics
            break
        elif hasattr(curr_env, 'env'):
            curr_env = curr_env.env
        else:
            break

    if physics is None:
        raise AttributeError("CRITICAL: Could not find the dm_control physics engine.")

    for i in range(num_states):
        # Safely teleport the cheetah
        with physics.reset_context():
            physics.named.data.qpos['rootz'] = CONSTANT_HEIGHT
            physics.named.data.qpos['rooty'] = angles[i].item()
            physics.data.qvel[:] = 0

        physics.forward()

        # Render and stack
        frame = physics.render(height=64, width=64, camera_id=0)
        frame_tensor = torch.tensor(frame.copy(), dtype=torch.float32, device=device).permute(2, 0, 1)
        stacked_obs = frame_tensor.repeat(3, 1, 1)
        obs_list.append(stacked_obs)

    obs_batch = torch.stack(obs_list)
    print(f"Created image batch of shape: {obs_batch.shape}")

    # Zero Action
    action_dim = env.action_space.shape[0]
    action_batch = torch.zeros((num_states, action_dim), device=device)

    print(f"Evaluating {num_states} states through the world model...")

    with torch.no_grad():
        z = agent.model.encode(obs_batch, task=None)
        next_z, nu, alpha, beta = agent.model.next(z, action_batch, task=None)

        # Reward and Value
        reward_preds = agent.model.reward(z, action_batch, task=None)
        q_values = agent.model.Q(z, action_batch, task=None)

        if q_values.dim() == 3:
            value_preds = torch.min(q_values, dim=0)[0]
        else:
            value_preds = q_values

        reward_scalars = reward_preds.argmax(dim=-1).float().cpu().numpy()
        value_scalars = value_preds.argmax(dim=-1).float().cpu().numpy()

        # Calculate Mean of uncertainties across latent dims
        aleatoric_vec = aleatoric_uncertainty(alpha, beta)
        epistemic_vec = epistemic_uncertainty(nu, alpha, beta)

        aleatoric_mean = torch.mean(aleatoric_vec, dim=-1).cpu().numpy()

        # Extract raw evidence for the "Zoomed" relative grid
        raw_evidence = nu.detach().cpu().numpy()
        evidence_scalar = np.mean(raw_evidence, axis=-1)

        # Baseline subtraction for saturated epistemic signal
        zoomed_line = (evidence_scalar - np.min(evidence_scalar))
        if np.max(zoomed_line) < 1e-3:
            zoomed_line = zoomed_line * 1e5

        epistemic_line = zoomed_line
        epistemic_line = torch.mean(epistemic_vec, dim=-1).cpu().numpy()

    print("Evaluation complete.")

    # --- Plotting ---
    angles_np = angles.numpy()
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)

    # 1x2 grid of 2D plots
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot Aleatoric Line
    ax0.plot(angles_np, aleatoric_mean, color='tab:blue', linewidth=3, marker='', markersize=4)
    ax0.set_title(f'Aleatoric Uncertainty vs. Pitch (Height={CONSTANT_HEIGHT})')
    ax0.set_xlabel('Pitch Angle (rad)')
    ax0.set_ylabel('Uncertainty Magnitude')
    ax0.yaxis.set_major_formatter(formatter)
    ax0.grid(True, linestyle='--', alpha=0.7)

    # Plot Epistemic Line
    ax1.plot(angles_np, epistemic_line, color='tab:orange', linewidth=3, marker='', markersize=4)
    ax1.set_title('Epistemic Uncertainty vs. Pitch')
    ax1.set_xlabel('Pitch Angle (rad)')
    ax1.set_ylabel('Uncertainty Magnitude')
    ax1.yaxis.set_major_formatter(formatter)
    ax1.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return angles_np, reward_scalars, value_scalars, next_z

if __name__ == "__main__":
    evaluate_world_model_line()
