import sys
import os
import numpy as np
import gym
import gym_yumi 
import argparse
import pickle
from gym_yumi import envs
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.ddpg import NormalActionNoise
from stable_baselines import DDPG, HER
from stable_baselines.common import make_vec_env
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--new-name', default="ppo2_yumi", type=str, help='name of model to train')
    parser.add_argument('--old-name', default=None, type=str, help='name of model to train from')

    args = parser.parse_args()

    new_model_name = args.new_name
    old_model_name = args.old_name
    timesteps = 2e5
    save_freq = 1e4

    env = envs.GoalYumiEnv('left','peg_target_res',mode='passive', headless=True,maxval=0.1)
    if old_model_name:
        model = HER.load(os.path.join('./models/', old_model_name), env=env)
        #model.set_env(env)
        model.verbose = 1
        model.tensorboard_log = os.path.join('./tensorboard/her', new_model_name)
    else:
        n_actions = env.action_space.shape[0]
        noise_std = 0.2
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
        
        model = HER(MlpPolicy, env, DDPG, 
                    goal_selection_strategy=GoalSelectionStrategy.FUTURE,
                    policy_kwargs=dict(layers=[256, 256, 256]),
                    gamma=0.95, batch_size=256, buffer_size=1000000,
                    actor_lr=1e-3,critic_lr=1e-3, random_exploration=.3,
                    normalize_observations=True, action_noise=action_noise,
                    verbose=1, tensorboard_log=os.path.join('./tensorboard/her', new_model_name))
   
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=os.path.join("./models", new_model_name, "checkpoints/"))
    model.learn(total_timesteps=int(timesteps), callback = checkpoint_callback, 
                tb_log_name = new_model_name, log_interval=100)
    model.save(os.path.join('./models', new_model_name))

if __name__ == "__main__":
    main()
