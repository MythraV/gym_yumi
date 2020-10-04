import gym
import gym_yumi 
from gym_yumi import envs
import matplotlib.pyplot as plt
import numpy as np
import sys
from stable_baselines import DDPG, HER

def main(model_path, model_type):
    num_steps = 200

    if model_type == "ppo":
        env_type = envs.YumiEnv
    else:
        env_type = envs.GoalYumiEnv
    
    env = env_type('left','peg_target_res',mode='passive', headless=False,maxval=0.1)
 
    # in function imports to fix env problem, normally move to top of script
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines import PPO2, HER

    if model_type == "ppo":
        model = PPO2.load(model_path)
    else:
        model = HER.load(model_path, env=env)
    
    episode_rewards = [0.0]
    obs = env.reset()
    done = False
    for i in range(num_steps):
        # model.prection returns a tuple w/ (action, None) for some reason
        action = model.predict(obs)[0]
        #import pdb; pdb.set_trace()
        obs, reward, done, _ = env.step(action)
        episode_rewards.append(reward)
        if done:
            obs = env.reset()
            episode_rewards.append(0)
    fig, ax = plt.subplots()
    ax.plot(episode_rewards)
    plt.show()
    env.close()


if __name__ == "__main__":
    assert (len(sys.argv) == 3), "python test.py [model_path] [ppo/ddpg]"
    model_path = sys.argv[1]
    model_type = sys.argv[2]
    assert (model_type == "ppo" or model_type == "her"), "model must be her or ddpg"
    main(model_path, model_type)
    
 

