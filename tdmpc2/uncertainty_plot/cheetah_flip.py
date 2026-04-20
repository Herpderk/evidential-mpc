import torch
import numpy as np
import hydra
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import sys
from pathlib import Path
tdmpc2_path = Path(__file__).parent.parent
sys.path.append(str(tdmpc2_path))

from common.evidential import evidential_variance
from common.parser import parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2

# Line parameters (Only varying pitch angle)
CONSTANT_HEIGHT = 0.5  # Adjust this to your cheetah's typical resting height
ANGLE_RANGE = (-np.pi, np.pi)
RESOLUTION = 50

@hydra.main(config_name='config', config_path='..')
def evaluate_world_model_line(cfg: dict):
    """Loads the TD-MPC2 agent and evaluates density and variance across pitch angles."""
    # Ensure the task is set for cheetah
    cfg['task'] = 'cheetah-flip'
    cfg = parse_cfg(cfg)

    env = make_env(cfg)
    agent = TDMPC2(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
    agent.load(cfg.checkpoint)

    print(f"Generating pixel-based state line at constant height {CONSTANT_HEIGHT}...")

    # 1D array of pitch angles
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
            # rooty represents the pitch angle in dm_control's planar cheetah
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
        y = agent.model.simplicial2continuous(z)

        # Calculate density (log-prob) and variance
        evid_pred = agent.model.evidential_prediction(y, action_batch, task=None)
        density = agent.model.id_logprob(y, action_batch, task=None)
        var = evidential_variance(evid_pred)

        # Reward and Value predictions
        reward_preds = agent.model.reward(z, action_batch, task=None)
        q_values = agent.model.Q(z, action_batch, task=None)

        if q_values.dim() == 3:
            value_preds = torch.min(q_values, dim=0)[0]
        else:
            value_preds = q_values

        reward_scalars = reward_preds.argmax(dim=-1).float().cpu().numpy()
        value_scalars = value_preds.argmax(dim=-1).float().cpu().numpy()

    print("Evaluation complete.")

    # --- Plotting ---
    angles_np = angles.cpu().numpy()
    formatter = ScalarFormatter(useOffset=False)
    #formatter.set_scientific(False)

    # Plot Density Line (Square-ish Plot)
    fig_density = plt.figure(figsize=(8, 6))
    ax0 = fig_density.add_subplot(111)
    ax0.plot(angles_np, density.cpu().numpy(), color='tab:blue', linewidth=3, marker='o', markersize=4)
    ax0.set_title(f'Density vs. Pitch Angle\n(Height={CONSTANT_HEIGHT})')
    ax0.set_xlabel('Pitch Angle (rad)')
    ax0.set_ylabel('Density Log-Prob')
    #ax0.yaxis.set_major_formatter(formatter)
    ax0.grid(True, linestyle='--', alpha=0.7)
    fig_density.tight_layout()

    # Plot Variance Line (Square-ish Plot)
    fig_var = plt.figure(figsize=(8, 6))
    ax1 = fig_var.add_subplot(111)
    ax1.plot(angles_np, var.cpu().numpy(), color='tab:orange', linewidth=3, marker='o', markersize=4)
    ax1.set_title(f'Variance vs. Pitch Angle\n(Height={CONSTANT_HEIGHT})')
    ax1.set_xlabel('Pitch Angle (rad)')
    ax1.set_ylabel('Variance Magnitudes')
    ax1.set_yscale('log')
    #ax1.yaxis.set_major_formatter(formatter)
    ax1.grid(True, linestyle='--', alpha=0.7)
    fig_var.tight_layout()

    # Display both plots
    plt.show()

    return angles_np, reward_scalars, value_scalars, y

if __name__ == "__main__":
    evaluate_world_model_line()
