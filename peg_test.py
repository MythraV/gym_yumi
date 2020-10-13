import gym
import gym_yumi

import numpy as np

env_kwargs = {'headless':False, 'maxval': 1, 'random_peg':True, 'normal_offset':True}
env = gym.make('goal-yumi-pegtransfer-v0', **env_kwargs)

env.reset()
import pdb; pdb.set_trace()