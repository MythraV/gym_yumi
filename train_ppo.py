import gym
import gym_yumi 
from gym_yumi import envs
# for some reason this crashes if performed after the stable_baselines imports?
#env = make_vec_env('yumi-pegtransfer-v0')

num_cpu = 6

envs = [envs.YumiEnv('left','peg_target_res',mode='passive', headless=True,maxval=0.1) for i in range(num_cpu)]

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

def make_env(i, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = envs[i]
        env.seed(seed + i)
        return env
    set_global_seeds(seed)
    return _init

env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./tensorboard/ppo2/')
model.learn(total_timesteps=int(2e5))
model.save('ppo2_yumi')

'''
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
'''

'''
lst_o += oh.get_position()  # position
ValueError: operands could not be broadcast together with shapes (0,) (3,) 
'''