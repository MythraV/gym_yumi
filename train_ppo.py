import sys
import os
import numpy as np
import gym
import gym_yumi 
from gym_yumi import envs
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback


def main(new_model_name = "ppo2_yumi", old_model_name = None):
    num_cpu = 8
    timesteps = 2e6
    save_freq = 1e5
    def make_env(rank, seed=0):
        """
        Utility function for multiprocessed env.
        
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = envs.YumiEnv('left','peg_target_res',mode='passive', headless=True,maxval=0.1)
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    if old_model_name:
        model = PPO2.load(os.path.join('./models/', old_model_name))
        model.set_env(env)
        model.verbose = 1
        model.tensorboard_log = os.path.join('./tensorboard/ppo2', new_model_name)
    else:
        model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=os.path.join('./tensorboard/ppo2', new_model_name))
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=os.path.join("./models", new_model_name, "checkpoints"))
    model.learn(total_timesteps=int(timesteps), callback = checkpoint_callback)
    model.save(os.path.join('./models', new_model_name))

if __name__ == "__main__":
    # both models expected to be in the models folder
    # ex python train_ppo.py ppo2_yumi_2e6_2e5  ppo2_yumi_2e5
    assert (len(sys.argv) >= 2), "try: python train_ppo.py [new_model_name] [old_model_name](optional)"
    new_model_name = sys.argv[1]
    if (len(sys.argv) == 3):
        old_model_name = sys.argv[2]
    else:
        old_model_name = None
    main(new_model_name, old_model_name)
