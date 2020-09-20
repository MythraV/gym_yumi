import gym
import gym_yumi 
from gym_yumi import envs
import matplotlib.pyplot as plt
import numpy as np
import sys

def main(model_path):
    env = envs.YumiEnv('left','peg_target_res',mode='passive', headless=False,maxval=0.1)

    # in function imports to fix env problem, normally move to top of script
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines import PPO2
    model = PPO2.load(model_path)

    episode_rewards = [0.0]
    obs = env.reset()
    done = False
    while (not done):
        # model.prection returns a tuple w/ (action, None) for some reason
        action = model.predict(obs)[0]
        #import pdb; pdb.set_trace()
        obs, reward, done, info = env.step(action)
        episode_rewards.append(reward)
    fig, ax = plt.subplots()
    ax.plot(episode_rewards)
    plt.show()
    env.close()


if __name__ == "__main__":
    assert (len(sys.argv) == 2), "python test_ppo.py [model_path]"
    model_path = sys.argv[1]
    main(model_path)
    
 

