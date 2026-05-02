import torch
import numpy as np
import hydra
import os
import matplotlib.pyplot as plt

import sys
from pathlib import Path
tdmpc2_path = Path(__file__).parent.parent
sys.path.append(str(tdmpc2_path))

from common.evidential import evidential_variance
from common.parser import parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2

# 2D Grid Parameters for Reacher Joint Angles
JOINT1_RANGE = (-np.pi, np.pi)
JOINT2_RANGE = (-np.pi, np.pi)
RESOLUTION = 20  # Grid size (20x20 = 400 states rendered and evaluated)

@hydra.main(config_name='config', config_path='..')
def evaluate_world_model_landscape(cfg: dict):
    """Loads the TD-MPC2 agent and evaluates density and variance across a 2D grid of reacher joint angles."""
    # Defaulting to reacher-hard, but you can change this to reacher-hard if needed
    cfg['task'] = 'reacher-hard'
    cfg = parse_cfg(cfg)
    env = make_env(cfg)
    agent = TDMPC2(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
    agent.load(cfg.checkpoint)
    agent.model.eval()

    print(f"Generating 2D pixel-based state grid (Resolution {RESOLUTION}x{RESOLUTION})...")

    # Create 2D grid for Joint 1 and Joint 2 angles
    j1_vals = np.linspace(JOINT1_RANGE[0], JOINT1_RANGE[1], RESOLUTION)
    j2_vals = np.linspace(JOINT2_RANGE[0], JOINT2_RANGE[1], RESOLUTION)
    J1, J2 = np.meshgrid(j1_vals, j2_vals)
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
    flat_j1 = J1.flatten()
    flat_j2 = J2.flatten()

    # Capture the initial target position so it stays constant during evaluation
    target_x = physics.named.model.geom_pos['target', 'x']
    target_y = physics.named.model.geom_pos['target', 'y']

    for i in range(num_states):
        # Safely teleport the reacher joints while keeping target fixed
        with physics.reset_context():
            # Zero out velocities
            physics.data.qvel[:] = 0
            physics.data.qpos[0] = flat_j1[i]
            physics.data.qpos[1] = flat_j2[i]

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
        y = agent.model.simplicial2continuous(z)

        # Calculate density (log-prob) and variance
        evid_pred = agent.model.evidential_prediction(y, action_batch, task=None)
        density = agent.model.id_logprob(y, action_batch, task=None)
        var = evidential_variance(evid_pred)

        # Reward and Value predictions (Kept for return values)
        reward_preds = agent.model.reward(z, action_batch, task=None)
        q_values = agent.model.Q(z, action_batch, task=None)

        if q_values.dim() == 3:
            value_preds = torch.min(q_values, dim=0)[0]
        else:
            value_preds = q_values

        reward_scalars = reward_preds.argmax(dim=-1).float().cpu().numpy()
        value_scalars = value_preds.argmax(dim=-1).float().cpu().numpy()

    print("Evaluation complete. Plotting 3D landscapes...")

    # Reshape the flat 1D output vectors back into 2D arrays matching the grid resolution
    density_2d = density.cpu().numpy().reshape(RESOLUTION, RESOLUTION)
    var_2d = torch.sum(var, dim=-1).cpu().numpy().reshape(RESOLUTION, RESOLUTION)
    reward_2d = reward_scalars.reshape(RESOLUTION, RESOLUTION)
    value_2d = value_scalars.reshape(RESOLUTION, RESOLUTION)

    # --- Plotting 3D Surface Landscapes ---
    fig = plt.figure(figsize=(16, 7))

    # Plot Density 3D Landscape
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    surf0 = ax0.plot_surface(J1, J2, density_2d, cmap='viridis', edgecolor='none', alpha=0.9)
    ax0.set_title(f'Density (Log-Prob) Variation\n(Target Fixed at X={target_x:.2f}, Y={target_y:.2f})', pad=15)
    ax0.set_xlabel('Joint 1 Angle (rad)', labelpad=10)
    ax0.set_ylabel('Joint 2 Angle (rad)', labelpad=10)
    ax0.set_zlabel('Density Log-Prob', labelpad=10)
    fig.colorbar(surf0, ax=ax0, shrink=0.5, aspect=10, pad=0.1, label='Density')

    # Plot Variance 3D Landscape
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    surf1 = ax1.plot_surface(J1, J2, var_2d, cmap='plasma', edgecolor='none', alpha=0.9)
    ax1.set_title(f'Variance Variation\n(Target Fixed at X={target_x:.2f}, Y={target_y:.2f})', pad=15)
    ax1.set_xlabel('Joint 1 Angle (rad)', labelpad=10)
    ax1.set_ylabel('Joint 2 Angle (rad)', labelpad=10)
    ax1.set_zlabel('Variance Magnitude', labelpad=10)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1, label='Variance')

    # Adjust view angles for better visibility if needed
    ax0.view_init(elev=30, azim=-45)
    ax1.view_init(elev=30, azim=-45)

    plt.tight_layout()
    plt.show()

    return J1, J2, reward_2d, value_2d, y

if __name__ == "__main__":
    evaluate_world_model_landscape()
