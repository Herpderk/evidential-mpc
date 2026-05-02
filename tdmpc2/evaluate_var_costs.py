import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import hydra
import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
from termcolor import colored
from tqdm import tqdm  # <-- Imported tqdm here

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


def compute_95_ci(data):
    """Computes the mean and 95% confidence interval of a list of values."""
    n = len(data)
    if n <= 1:
        return np.mean(data), np.mean(data), np.mean(data)
    m = np.mean(data)
    # Using 1.96 for a 95% confidence interval
    se = np.std(data, ddof=1) / np.sqrt(n)
    h = 1.96 * se
    return m, m - h, m + h


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
    """
    Script for evaluating a single-task / multi-task TD-MPC2 checkpoint
    across varying `var_cost_coef` values.
    """
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
    print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
    print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
    if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
        print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
        print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

    # Make environment
    env = make_env(cfg)

    # Load agent
    agent = TDMPC2(cfg)
    assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
    agent.load(cfg.checkpoint)

    # Evaluate
    if cfg.multitask:
        print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
    else:
        print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))

    if cfg.save_video:
        video_dir = os.path.join(cfg.work_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)

    #plot_dir = os.path.join(cfg.work_dir, 'plots')
    plot_dir = '/home/user3/Documents/evidential-mpc-outputs/eval_epochs'
    os.makedirs(plot_dir, exist_ok=True)

    tasks = cfg.tasks if cfg.multitask else [cfg.task]

    # Range of coefficients to test: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    coef_list = np.arange(0.0, 0.6, 0.1) #[0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    print('starting eval...')
    for task_idx, task in enumerate(tasks):
        if not cfg.multitask:
            task_idx = None

        task_means = []
        task_lower_cis = []
        task_upper_cis = []

        print(colored(f'--- Task: {task} ---', 'magenta', attrs=['bold']))

        for coef in coef_list:
            # Update agent config before evaluation
            agent.cfg.use_var_cost = True
            agent.cfg.var_cost_coef = float(coef)

            ep_rewards, ep_successes = [], []

            # <-- Added tqdm progress bar here
            for i in tqdm(range(cfg.eval_episodes), desc=f"Eval coef {coef:.1f}"):
                obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
                if cfg.save_video:
                    frames = [env.render()]
                while not done:
                    action = agent.act(obs, t0=t==0, task=task_idx)
                    obs, reward, done, info = env.step(action)
                    ep_reward += reward
                    t += 1
                    if cfg.save_video:
                        frames.append(env.render())

                ep_rewards.append(ep_reward)
                ep_successes.append(info['success'])

                if cfg.save_video:
                    imageio.mimsave(
                        os.path.join(video_dir, f'{task}-coef{coef:.1f}-{i}.mp4'), frames, fps=15)

            # Compute Statistics
            mean_r, lower_r, upper_r = compute_95_ci(ep_rewards)
            mean_success = np.mean(ep_successes)

            task_means.append(mean_r)
            task_lower_cis.append(lower_r)
            task_upper_cis.append(upper_r)

            print(colored(f'  coef: {coef:.1f}' \
                f'\tR: {mean_r:.01f} (95% CI: [{lower_r:.01f}, {upper_r:.01f}])  ' \
                f'\tS: {mean_success:.02f}', 'yellow'))

        # Plotting the results for the current task
        plt.figure(figsize=(8, 6))
        plt.plot(coef_list, task_means, marker='o', color='b', label='Mean Reward')
        plt.fill_between(coef_list, task_lower_cis, task_upper_cis, color='b', alpha=0.2, label='95% Confidence Interval')

        plt.xlabel('var_cost_coef', fontsize=12)
        plt.ylabel('Episode Reward', fontsize=12)
        plt.title(f'Performance vs Variance Cost Coefficient\nTask: {task}', fontsize=14)
        plt.xticks(coef_list)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        plot_path = os.path.join(plot_dir, f'{task}_var_cost_eval.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(colored(f'  Saved plot to {plot_path}', 'green'))

if __name__ == '__main__':
    evaluate()
