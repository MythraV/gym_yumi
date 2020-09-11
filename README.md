The gym environment for Yumi

# gym-yumi

gym environment for ABB Yumi robot using VREP simulator.

## Requirements

Please install the following requirements before using this environment. It is recommended to do this in a virtual environment..

Python 3.8.5
[VREP](https://www.coppeliarobotics.com/previousVersions) version 4.1.0
[PyRep](https://github.com/MythraV/PyRep.git) version 1.2
[gym](https://github.com/openai/gym.git) version 0.17.2


## RL setup


# Installation

```bash
git clone https://github.com/MythraV/gym_yumi.git
cd gym-yumi
pip install -e .
```

# Example Usage
Test environment with gym
```bash
python
import gym
import gym_yumi
env = gym.make('yumi-pegtransfer-v0')
```
Running with openai/baselines or stable_baselines
```bash
python -m baselines.run --alg=ppo2 --env=gym_yumi:yumi-pegtransfer-v0 --network=mlp --num_timesteps=2e7 --num_env=6 --save_path=/media/crl/DATA/Datasets/RLmodels/yumi --save_interval=1e5
```
