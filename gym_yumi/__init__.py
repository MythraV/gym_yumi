from gym.envs.registration import register

register(
    id='yumi-pegtransfer-v0',
    entry_point='gym_yumi.envs:YumiEnv',
    reward_threshold=10.0,
    kwargs={'limb':'left','goal':'peg_left_1','headless':True,'maxval':0.1}
)

