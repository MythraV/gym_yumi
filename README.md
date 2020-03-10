The gym environment for Yumi

# gym-yumi

gym environment for ABB Yumi robot using VREP simulator. 

## Requirements

Please install the following requirements before using this environment. It is recommended to do this in a virtual environment..

[VREP](https://www.coppeliarobotics.com/previousVersions) version 3.6.2
[PyRep](https://github.com/MythraV/PyRep.git) version compatible with VREP 3.6
[gym](https://github.com/openai/gym.git) 

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
env = gym.make('gym-yumi:yumi-pegtransfer-v0')
