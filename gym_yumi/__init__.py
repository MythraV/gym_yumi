from gym.envs.registration import register

register(
    id='yumi-pegtransfer-v0',
    entry_point='gym_yumi.envs:YumiEnv',
    reward_threshold=10.0,
<<<<<<< HEAD
    kwargs={'limb':'left','headless':False,'maxval':0.1}
=======
    kwargs={'limb':'left','goals': ['approach'],'headless':False,'maxval':0.1}
>>>>>>> f03d9b25788760b865be722a3e28ffaa11b20f96
)

register(
    id='goal-yumi-pegtransfer-v0',
    entry_point='gym_yumi.envs:GoalYumiEnv',
    reward_threshold=10.0,
    kwargs={'limb':'left','headless':True,'maxval':0.1,
            'random_peg':True, 'normal_offset':True,
            'goals': ['approach'], 
            #'arm_configs': '/home/daniel/Projects/Python/gym_yumi/baselines/locations/her10_2.json',
            'random_peg_xy': False
            },
    max_episode_steps=50
)

register(
    id='grasp-goal-yumi-pegtransfer-v0',
    entry_point='gym_yumi.envs:GoalYumiEnv',
    reward_threshold=10.0,
    kwargs={'limb':'left','headless':True,'maxval':0.1,
            'random_peg':True, 'normal_offset':True,
            'goals': ['grasp'], 
            #'arm_configs': '/home/daniel/Projects/Python/gym_yumi/baselines/locations/her10_2.json',
            'random_peg_xy': True
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

register(
    id='yumi-pegtransfer-vec-v0',
    entry_point='gym_yumi.envs:SubprocVecYumiEnv',
    reward_threshold=10.0,
    kwargs={'limb':'left','headless':True,'maxval':0.1,
            'random_peg':True, 'normal_offset':True,'reward_name':'report_reward'
            },
    max_episode_steps=50
)
