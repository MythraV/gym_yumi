from gym_yumi.envs import GoalYumiEnv
from gym_yumi.envs.subproc_vec_env import SubprocVecEnv

import gym
from gym import spaces

class SubprocVecYumiEnv(gym.Env):
  def __init__(self, **kwargs):
    def create_env():
      return GoalYumiEnv(**kwargs)
    
    self.env = SubprocVecEnv([create_env])

    self.action_space = self.env.get_attr("action_space")[0]
    self.observation_space = self.env.get_attr("observation_space")[0]

  def step(self, action):
    return self.env.step([action])
    
  def reset(self):
    return self.env.reset()

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    pass