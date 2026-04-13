import torch
import numpy as np
import hydra
import os
import matplotlib.pyplot as plt

import sys
from pathlib import Path
tdmpc2_path = Path(__file__).parent.parent
sys.path.append(str(tdmpc2_path))

from common.evidential import aleatoric_uncertainty, epistemic_uncertainty
from common.parser import parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2

# 2D Grid Parameters for Ball-in-Cup
CUP_X = 0.0
CUP_Z = 0.0
BALL_X_RANGE = (-0.5, 0.5)
BALL_Z_RANGE = (-0.5, 0.5)
RESOLUTION = 20  # Grid size (20x20 = 400 states rendered and evaluated)

@hydra.main(config_name='config', config_path='..')
def evaluate_world_model_landscape(cfg: dict):
    """Loads the TD-MPC2 agent and evaluates uncertainty across a 2D grid of ball positions."""
    cfg['task'] = 'cup-catch'
    cfg = parse_cfg(cfg)
    env = make_env(cfg)
    agent = TDMPC2(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
    agent.load(cfg.checkpoint)

    print(f"Generating 2D pixel-based state grid (Resolution {RESOLUTION}x{RESOLUTION})...")

    # Create 2D grid for Ball X and Z positions
    x_vals = np.linspace(BALL_X_RANGE[0], BALL_X_RANGE[1], RESOLUTION)
    z_vals = np.linspace(BALL_Z_RANGE[0], BALL_Z_RANGE[1], RESOLUTION)
    X, Z = np.meshgrid(x_vals, z_vals)
    num_states = RESOLUTION * RESOLUTION

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

    # Flatten the grid arrays to iterate through all combinations easily
    flat_x = X.flatten()
    flat_z = Z.flatten()

    for i in range(num_states):
        # Safely teleport the cup and the ball
        with physics.reset_context():
            # Zero out velocities
            physics.data.qvel[:] = 0

            # In standard dm_control ball_in_cup, qpos has 4 coordinates:
            # Index 0: cup_x, Index 1: cup_z, Index 2: ball_x, Index 3: ball_z
            physics.data.qpos[0] = CUP_X
            physics.data.qpos[1] = CUP_Z
            physics.data.qpos[2] = flat_x[i]
            physics.data.qpos[3] = flat_z[i]

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
        next_z, nu, alpha, beta = agent.model.next_noise(z, action_batch, task=None)

        # Reward and Value predictions
        reward_preds = agent.model.reward(z, action_batch, task=None)
        q_values = agent.model.Q(z, action_batch, task=None)

        if q_values.dim() == 3:
            value_preds = torch.min(q_values, dim=0)[0]
        else:
            value_preds = q_values

        reward_scalars = reward_preds.argmax(dim=-1).float().cpu().numpy()
        value_scalars = value_preds.argmax(dim=-1).float().cpu().numpy()

        # Calculate Mean of uncertainties across latent dims
        aleatoric_vec = aleatoric_uncertainty(nu, alpha, beta)
        epistemic_vec = epistemic_uncertainty(nu)

        aleatoric_mean = torch.mean(aleatoric_vec, dim=-1).cpu().numpy()

        # Use normalized/zoomed epistemic logic or default mean
        raw_evidence = nu.detach().cpu().numpy()
        evidence_scalar = np.mean(raw_evidence, axis=-1)
        zoomed_line = (evidence_scalar - np.min(evidence_scalar))

        if np.max(zoomed_line) < 1e-3:
            zoomed_line = zoomed_line * 1e5

        # We fall back to standard uncertainty measure for robustness,
        # but you can swap to zoomed_line if needed.
        epistemic_mean = torch.mean(epistemic_vec, dim=-1).cpu().numpy()

    print("Evaluation complete. Plotting 3D landscapes...")

    # Reshape the flat 1D output vectors back into 2D arrays matching the grid resolution
    aleatoric_2d = aleatoric_mean.reshape(RESOLUTION, RESOLUTION)
    epistemic_2d = epistemic_mean.reshape(RESOLUTION, RESOLUTION)
    reward_2d = reward_scalars.reshape(RESOLUTION, RESOLUTION)
    value_2d = value_scalars.reshape(RESOLUTION, RESOLUTION)

    # --- Plotting 3D Surface Landscapes ---
    fig = plt.figure(figsize=(16, 7))

    # Plot Aleatoric 3D Landscape
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    surf0 = ax0.plot_surface(X, Z, aleatoric_2d, cmap='viridis', edgecolor='none', alpha=0.9)
    ax0.set_title(f'Aleatoric Uncertainty\n(Cup Fixed at X={CUP_X}, Z={CUP_Z})', pad=15)
    ax0.set_xlabel('Ball X Position', labelpad=10)
    ax0.set_ylabel('Ball Z Position', labelpad=10)
    ax0.set_zlabel('Uncertainty Magnitude', labelpad=10)
    fig.colorbar(surf0, ax=ax0, shrink=0.5, aspect=10, pad=0.1, label='Uncertainty')

    # Plot Epistemic 3D Landscape
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    surf1 = ax1.plot_surface(X, Z, epistemic_2d, cmap='plasma', edgecolor='none', alpha=0.9)
    ax1.set_title(f'Epistemic Uncertainty\n(Cup Fixed at X={CUP_X}, Z={CUP_Z})', pad=15)
    ax1.set_xlabel('Ball X Position', labelpad=10)
    ax1.set_ylabel('Ball Z Position', labelpad=10)
    ax1.set_zlabel('Uncertainty Magnitude', labelpad=10)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1, label='Uncertainty')

    # Adjust view angles for better visibility if needed
    ax0.view_init(elev=30, azim=-45)
    ax1.view_init(elev=30, azim=-45)

    plt.tight_layout()
    plt.show()

    return X, Z, reward_2d, value_2d, next_z

if __name__ == "__main__":
    evaluate_world_model_landscape()
