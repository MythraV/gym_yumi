from gym.envs.registration import register

register(
    id='yumi-pegtransfer-v0',
    entry_point='gym_yumi.envs:YumiEnv',
    reward_threshold=10.0,
    kwargs={'limb':'left','goal':'peg_target_res','headless':False,'maxval':0.1}
)

register(
    id='goal-yumi-pegtransfer-v0',
    entry_point='gym_yumi.envs:GoalYumiEnv',
    reward_threshold=10.0,
    kwargs={'limb':'left','headless':True,'maxval':0.1,
            'random_peg':True, 'normal_offset':True,
            'goals': ['grasp'], 
            'arm_configs': '/home/daniel/Projects/Python/gym_yumi/baselines/locations/her10_2.json'
            },
    max_episode_steps=50
)

register(
    id='goal-original-yumi-pegtransfer-v0',
    entry_point='gym_yumi.envs:GoalYumiEnv',
    reward_threshold=10.0,
    kwargs={'limb':'left','headless':True,'maxval':0.1,
            'random_peg':True, 'normal_offset':True,
            },
    max_episode_steps=50
)