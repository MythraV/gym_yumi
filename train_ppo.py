import gym
import gym_yumi 

# for some reason this crashes if performed after the stable_baselines imports?
env = gym.make('yumi-pegtransfer-v0')

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2


model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=int(2e7))


obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()

