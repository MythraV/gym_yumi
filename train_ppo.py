import gym
import gym_yumi 
from gym_yumi import envs
# for some reason this crashes if performed after the stable_baselines imports?
# env = gym.make('yumi-pegtransfer-v0')

env = envs.YumiEnv('left','peg_target_res',mode='passive', headless=True,maxval=0.1)

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2


model = PPO2(MlpPolicy, env, verbose=1)
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