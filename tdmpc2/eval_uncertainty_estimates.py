import torch
import numpy as np
import hydra
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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
    """Loads the TD-MPC2 agent and evaluates OOD logprob across pitch angles."""
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
        # Encode observations into latents
        z = agent.model.encode(obs_batch, task=None)

        # Predict the next latent state (we only need the first output, next_z)
        z_next = agent.model.next(z, action_batch, task=None)

        # Compute the Out-Of-Distribution Log Probability
        logprobs = agent.model.ood_logprob(z_next, z, action_batch, task=None)

        # Move to CPU and numpy for plotting
        # (Assuming logprob returns a scalar per state in the batch)
        logprobs_np = logprobs.squeeze().cpu().numpy()

    print("Evaluation complete.")

    # --- Plotting ---
    angles_np = angles.numpy()
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)

    # Single plot for the OOD Logprob
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(angles_np, logprobs_np, color='tab:red', linewidth=3, marker='o', markersize=4)
    ax.set_title(f'OOD Log-Probability vs. Pitch (Height={CONSTANT_HEIGHT})')
    ax.set_xlabel('Pitch Angle (rad)')
    ax.set_ylabel('Log-Probability')
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return angles_np, logprobs_np, next_z

if __name__ == "__main__":
    evaluate_world_model_line()
