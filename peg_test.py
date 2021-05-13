import gym
import gym_yumi

import numpy as np

if __name__ == "__main__":
    env_kwargs = {'headless':False, 'reward_name':'report_reward'}#, 'maxval': 1, 'random_peg':True, 'normal_offset':True}
    env = gym.make('yumi-pegtransfer-vec-v0', **env_kwargs)

    env.reset()
    import pdb; pdb.set_trace()
