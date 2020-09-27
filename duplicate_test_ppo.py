import gym
import gym_yumi 
from gym_yumi import envs
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pickle

def main():
    num_steps = 20
    env = envs.YumiEnv('left','peg_target_res',mode='passive', headless=False,maxval=0.1)

    # in function imports to fix env problem, normally move to top of script
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines import PPO2

    model = PPO2(MlpPolicy, env, verbose=1)
    reacher_parameters = pickle.load(open('./models/parameters_reacher.pkl', 'rb'))
    reacher_dictionary = reacher_parameters[1]

    load_dictionary = {}

    layers_to_copy = [
        'model/pi_fc1/w:0', # (64, 64)
        'model/pi_fc1/b:0', # (64,)

        'model/vf_fc1/w:0', # (64, 64)
        'model/vf_fc1/b:0', # (64,)
        'model/vf/w:0',     # (64,)
        'model/vf/b:0',     # (64, 1)
    ]

    for layer in layers_to_copy:
        load_dictionary[layer] = reacher_dictionary[layer]

    model.load_parameters(load_dictionary, exact_match=False)
    model.setup_model()

    episode_rewards = [0.0]
    obs = env.reset()
    done = False
    for i in range(num_steps):
        # model.prection returns a tuple w/ (action, None) for some reason
        action = model.predict(obs)[0]
        #import pdb; pdb.set_trace()
        obs, reward, done, info = env.step(action)
        episode_rewards.append(reward)
        if done:
            obs = env.reset()
            episode_rewards.append(0)
    fig, ax = plt.subplots()
    ax.plot(episode_rewards)
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
    
 

