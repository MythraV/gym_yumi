from gym_yumi.envs.robot import *
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
import time
import gym
from gym import spaces, GoalEnv
from collections import OrderedDict
import numpy as np
import os
import random
import math
import json


class Yumi():
    def __init__(self):
        joint_names = {'left':[
            'yumi_l_joint1','yumi_l_joint2','yumi_l_joint3','yumi_l_joint4',
            'yumi_l_joint5','yumi_l_joint6','yumi_l_joint7'
        ],
        'right': [
            'yumi_r_joint1','yumi_r_joint2','yumi_r_joint3','yumi_r_joint4',
            'yumi_r_joint5','yumi_r_joint6','yumi_r_joint7'
        ]}
        self.left = Limb('yumi_l_tip', joint_names['left'])
        self.left_gripper = TwoFingerGripper(['gripper_l_joint','gripper_l_joint_m'])
        self.right_gripper = TwoFingerGripper(['gripper_r_joint','gripper_r_joint_m'])
        self.right_gripper.open()
        self.left_gripper.open()
        self.left_open = True
        self.right_open = True

# goals: approach, grasp

class YumiEnv(gym.Env):
    def __init__(self, limb='left', goals=['approach'], reward_name='original_reward', headless=False, mode='passive', maxval=0.1, 
                 normal_offset = False, random_peg = False, SCENE_FILE = None, arm_configs = None, 
                 terminal_distance=0.02, lift_height = 0.05, random_peg_xy = False, 
                 add_grasp_reward=False, physics_on=True):
        self.pr = PyRep()
        if SCENE_FILE is None:
            SCENE_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yumi_setup.ttt')
        # SCENE_FILE = '/homes/lawson95/CRL/gym_yumi/gym_yumi/envs/yumi_setup.ttt'
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()
        
        yumi = Yumi()
        if limb=='left':
            self.limb = yumi.left
            self.gripper = yumi.left_gripper
            self.limb_name = 'left'
        elif limb=='right':
            self.limb = yumi.right
            self.gripper = yumi.right_gripper
            self.limb_name = 'right'

        self.mode = mode
        if mode=='force':
            self.limb.set_joint_mode('force')
        shape_names = ['peg_target_res']

        # Relevant scene objects
        self.oh_shape = [Shape(x) for x in shape_names]
        self.peg_target = self.oh_shape[0]
        #print("oh_shape 1", self.oh_shape)
        # Add tool tip
        self.oh_shape.append(self.limb.target)
        #print("oh_shape 2", self.oh_shape)

        # Number of actions
        num_act = len(self.limb.joints)
    

        # Observation space size
        # 6 per object (xyzrpy) + 6 dimensions
        num_obs_obj = len(self.oh_shape)*3*2
        # Setup action and observation spaces
        act = np.array( [maxval] * num_act )
        obs = np.array(          [np.inf]          * num_obs_obj )

        # add to obs/act space if grasp for gripper
        if goals[0] == 'grasp':
            self.action_space      = spaces.Box(np.append(-act,0) , np.append(act, 1))
            self.observation_space = spaces.Box(np.append(-obs, 0), np.append(obs, 1))
        else:
            self.action_space      = spaces.Box(-act,act)
            self.observation_space = spaces.Box(-obs,obs)

        self.random_peg = random_peg
        self.normal_offset = normal_offset
        self.pegs = ['peg_left_{}'.format(i + 1) for i in range(6)]
        self.pegs.extend(['peg_right_{}'.format(i + 1) for i in range(6)])
        self.peg_name = None
        self.reward_name = reward_name
        self.rewardfcn = self._get_reward
        self.terminal_distance = terminal_distance
        self.lift_height = lift_height
        self.target_peg_pos = None
        self.add_grasp_reward = add_grasp_reward
        self.physics_on = True


        valid_goals = ['approach','grasp']
        for goal in goals:
            assert (goal in valid_goals), "Invalid goal, check valid_goals"
        self.goals = goals

        self.default_config = [-1.0122909545898438, -1.5707963705062866,
                             0.9759880900382996, 0.6497860550880432,
                              1.0691887140274048, 1.1606439352035522,
                               0.3141592741012573]

        self.grasp_config = [-1.2927653769521998, -1.4086552392846232, 1.1728612573401895,
                             -0.42149701435663056, 0.8768534162019512, 1.4494959437812907, 
                            1.0538298023541763]

        # load dictionary, containing array of joint configs for each peg position
        if arm_configs == None:
            self.arm_configs = None
        else:
            with open(arm_configs) as load_file:
                self.arm_configs = json.load(load_file)
        self._max_episode_steps = 50
        
        # parameter for random peg xy
        self.random_peg_xy = random_peg_xy
        self.random_peg_xy_run = False
        self.peg_xyz = [3.2516e-1, 7.6894e-2, 1.0218e0]
        self.peg_xy_offset = {
            "x_min": -0.05,
            "x_max": 0.05,
            "y_min": -0.05,
            "y_max": 0.05,
        }
        self.yumi_body = Shape('yumi_body_respondable')
        self.yumi_body_config = self.yumi_body.get_configuration_tree()
        self.yumi_peg_config = Shape('peg_target_res').get_configuration_tree()

    def get_random_peg_xyz(self):
        x_offset = random.random() * (self.peg_xy_offset['x_max'] - self.peg_xy_offset['x_min']) + self.peg_xy_offset['x_min']
        y_offset = random.random() * (self.peg_xy_offset['y_max'] - self.peg_xy_offset['y_min']) + self.peg_xy_offset['y_min']
        return [self.peg_xyz[0] + x_offset, self.peg_xyz[1] + y_offset, self.peg_xyz[2]]


    def _make_observation(self):
        """Query V-rep to make observation.
            The observation is stored in self.observation
        """
        lst_o = []
        # example: include position, linear and angular velocities of all shapes
        for oh in self.oh_shape:
            lst_o.extend(oh.get_position()) 	# position
            lst_o.extend(oh.get_orientation())
        if self.goals[0] == 'grasp':
            lst_o.append(self.gripper.get_open())
        self.observation = np.array(lst_o).astype('float32')

    def _make_action(self, actions):
        """Query V-rep to make action.
           no return value
        """
        joint_actions = actions
        if self.goals[0] == 'grasp':
            # map continuous action (0 - 1 to discrete control)
            if actions[-1] > .5:
                self.gripper.open()
            else:
                self.gripper.close()
            joint_actions = actions[1:]

        if self.mode=='force':
            # example: set a velocity for each joint
            for jnt, act in enumerate(joint_actions):
                self.limb.set_joint_velocity(jnt, act)
        else:
            # example: set a offset for each joint
            for jnt, act in enumerate(joint_actions):
                self.limb.offset_joint_position(jnt, act)

    def _get_distance(self, achieved_goal = None, desired_goal = None):
        if type(achieved_goal) == type(None):
            achieved_goal = self.observation[6:9]
        if type(desired_goal) == type(None):
            desired_goal = self.observation[0:3]
        return np.linalg.norm(desired_goal - achieved_goal)

    def _done(self):
        if self._get_distance() < self.terminal_distance:
            return True
        return False

    def step(self, action):
        """Gym environment 'step'
        """
        # Assert the action space space contains the action
        assert self.action_space.contains(action), "Action {} ({}) is invalid".format(action, type(action))
        
        # make sure phyics on
        if not self.pr.running:
            self.pr.start()
            
        # Actuate
        self._make_action(action)
        # Step
        self.pr.step()
        # Observe
        self._make_observation()

        # Reward
        reward = self.rewardfcn()

        # Early stop
        # if the episode should end earlier
        # done = if position outside user model space
        # modified to always run until max timesteps
        done = False
        info = {
            'is_success': self._done(),
        }
        return self.observation, reward, done, info

    def reset(self):
        """Gym environment 'reset'
        """
        # if self.physics_on:
        #     # modified restart
        #     if not self.pr.running:
        #         self.pr.start()
        #     #self.pr.start()
        #     self.pr.stop()
        #     self.pr.start()
        #     self.pr.step()
        #     self.pr.step()
        # else:
        #     if self.pr.running:
        #         self.pr.stop()
        
        if not self.pr.running:
             self.pr.start()

        # reset body and peg
        self.pr.set_configuration_tree(self.yumi_body_config)
        self.pr.set_configuration_tree(self.yumi_peg_config)


        # If approach and random_peg_xy, then randomly put peg in xy plane and have goal be in the same xy plane
        self.random_peg_xy_run = False
        if self.goals[0] == 'grasp':
            if self.random_peg_xy:
                if random.random() > .5:
                    self.random_peg_xy_run = True
        
        peg_position = [2.9878e-01, 5.1591e-02, 1.0201e+00]
        
        # get which peg the object is placed on
        self.peg_name = 'peg_left_2'
        if self.random_peg:
            if self.random_peg_xy_run:
                # get a random xy (and a set z)
                new_pos = self.get_random_peg_xyz()
                Shape('peg_target_res').set_position(new_pos)
            else:
                # get a random peg position
                self.peg_name = self.pegs[random.randrange(len(self.pegs))]
                pos = Shape(self.peg_name).get_position()
                # update peg position
                Shape('peg_target_res').set_position(pos)
                #Shape('peg_target').set_position(pos)
                #Shape('peg_target').rotate((0,0, random.random() * 2 * math.pi))
        elif self.goals[0] == 'grasp':
            Shape('peg_target_res').set_position(peg_position)
        
        grasp_points = [Dummy('grasp_point1').get_position(), Dummy('grasp_point2').get_position(), Dummy('grasp_point3').get_position()]

        # set up env for goal

        self.limb.set_joint_mode(self.mode)
        if self.goals[0] == 'approach':
            for i in range(len(self.limb.joints)):
                offset = 0
                if self.normal_offset:
                    offset = np.random.normal(0, .05)
                self.limb.set_joint_position(i,self.default_config[i] + offset)
            self.target_peg_pos = None

        elif self.goals[0] == 'grasp':            
            # if self.random_peg_xy_run:
            #     put arm in original position
            #     for i in range(len(self.limb.joints)):
            #         offset = 0
            #         if self.normal_offset:
            #             offset = np.random.normal(0, .05)
            #         self.limb.set_joint_position(i,self.default_config[i] + offset)
                
            #     # put arm next to peg

            # else:
            #     # get a random config for the peg
            #     num_configs = len(self.arm_configs[self.peg_name])
            #     joint_config = self.arm_configs[self.peg_name][random.randrange(num_configs)]
            #     # update joint positions
            #     for i in range(len(self.limb.joints)):
            #         self.limb.set_joint_position(i, joint_config[i])
            
            # put end effector next to peg
            #self.limb.target.set_position(grasp_points[random.randrange(3)])
            for i in range(len(self.limb.joints)):
                self.limb.set_joint_position(i,self.grasp_config[i])


            if self.random_peg_xy_run:
                self.gripper.open()
        
        # if self.physics_on:
        #     if not self.pr.running:
        #         self.pr.start()
        #         self.pr.step()
        #         self.pr.step()
        
        self._make_observation()

        if not self.pr.running:
             self.pr.start()

        return self.observation

    def render(self, mode='human', close=False):
        """Gym environment 'render'
        """
        pass

    def seed(self, seed=None):
        """Gym environment 'seed'
        """
        return []

    def close(self):
        """
            Shutdown function
        """
        self.pr.stop()
        self.pr.shutdown()

    def stop(self):
        '''
            Stop simulation
        '''
        self.pr.stop()

    def start(self):
        '''
            Start simulation
        '''
        self.pr.start()

    def _get_reward(self):
        '''
            The reward function
            Put your reward function here
        '''
        if self.goals[0] == 'grasp':
            achieved_goal = Shape('peg_target_res').get_position()
            desired_goal = self.target_peg_pos
            distance = self._get_distance(achieved_goal=achieved_goal, desired_goal=desired_goal)
        else:
            distance = self._get_distance() # approach
        return getattr(self,self.reward_name)(distance)

    def original_reward(self, distance):
        return 1.0 - 2 * distance

    def report_reward(self, distance):
        if distance < self.terminal_distance:
            reward = 10
        else:
            reward = -2 * abs(self.terminal_distance - 2 * distance)

        return reward

    def sparse_reward(self, distance):
        if distance < self.terminal_distance:
            reward = 0
        else:
            reward = -1
        return reward



class GoalYumiEnv(YumiEnv, GoalEnv):
    def __init__(self, *args, **kwargs):
        YumiEnv.__init__(self, *args, **kwargs)
        self.reward_name = 'sparse_reward'
        self.observation_space = spaces.Dict({
            'observation': self.observation_space,
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(3,)),
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(3,))
        })
        self.grasped_once = False

    def reset(self):
        """Gym environment 'reset'
        """
        observation = YumiEnv.reset(self)
        # self.grasped_once = True? Disable?
        if self.add_grasp_reward:
            self.grasped_once = False
        else:
            self.grasped_once = True
        return observation

    def _make_observation(self):
        """
        Helper to create the observation.
        :return: (OrderedDict<int or ndarray>)
        """
        YumiEnv._make_observation(self)

        # change goal depending on grasp or approach
        if self.goals[0] == 'grasp':
            if self.grasped_once:
                desired_goal = self.target_peg_pos
                achieved_goal = Shape('peg_target_res').get_position()
            else:
                achieved_goal = self.observation[6:9]
                grasp_points = [Dummy('grasp_point1').get_position(), Dummy('grasp_point2').get_position(), Dummy('grasp_point3').get_position()]
                min_distance = float('inf')
                distances = []
                min_index = -1
                for i in range(len(grasp_points)):
                    distance = self._get_distance(achieved_goal=achieved_goal, desired_goal=grasp_points[i])
                    distances.append(distance)
                    if distance < min_distance:
                        min_distance = distance
                        min_index = i
                desired_goal = grasp_points[min_index]
                
        else:
            desired_goal = self.observation[0:3]
            achieved_goal = self.observation[6:9]

        self.observation = OrderedDict([
            ('observation', self.observation),
            ('achieved_goal', achieved_goal),
            ('desired_goal', desired_goal)
        ])
    def _get_reward(self):
        '''
            The reward function
            Put your reward function here
        '''
        distance = self._get_distance()
        return getattr(self,self.reward_name)(distance)

    def _get_distance(self, achieved_goal = None, desired_goal = None):
        if type(achieved_goal) == type(None):
            achieved_goal = self.observation['achieved_goal']
        if type(desired_goal) == type(None):
            desired_goal = self.observation['desired_goal']
        return np.linalg.norm(desired_goal - achieved_goal)

    # additional argument, grasp toggle to trigger switching between training goal if goal is achieved
    def sparse_reward(self, distance, grasp_toggle = True):
        if distance < self.terminal_distance:
            reward = 0
            if grasp_toggle:
                self.grasped_once = True
        else:
            reward = -1
        return reward

    def compute_reward(self, achieved_goal, goal, info):
        return self.sparse_reward(self._get_distance(achieved_goal, goal), grasp_toggle = False)


if __name__ == "__main__":
    "Example usage for the YumiRLEnv"
    env = YumiEnv('left','peg_target_res',mode='passive', headless=True,maxval=0.1)
    print([env.limb.get_joint_position(x) for x in range(7)])
    time.sleep(1)
    for ieps in range(20):
        observation = env.reset()
        total_reward = 0
        action = env.action_space.sample()
        for t in range(5):
            #action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
            # time.sleep(0.1)
            # print(action[0], env.limb.get_joint_position(0))
        print("Episode {} finished after {} timesteps.\tTotal reward: {}".format(ieps+1,t+1,total_reward))
    env.close()
