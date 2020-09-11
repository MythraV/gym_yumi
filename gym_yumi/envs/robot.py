#from pyrep.backend.vrep import *
from os.path import dirname, join, abspath
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.const import JointMode

class TwoFingerGripper():
    def __init__(self, joints, open_pos=0.2):
        self.joints = [Joint(joints[0]),Joint(joints[1])]
        self.open_pos = open_pos

    def close(self):
        self.joints[0].set_joint_position(0)
        self.joints[1].set_joint_position(0)

    def open(self):
        self.joints[0].set_joint_position(self.open_pos)
        self.joints[1].set_joint_position(self.open_pos)

class Limb():
    def __init__(self, tip, joint_names):
        self.target = Dummy(tip)
        self.joints = [Joint(x) for x in joint_names]
        self.jointmode = 'position'

    def set_pose(self, pos, relative_to=None):
        # Set position
        self.target.set_position(pos[0:3],relative_to)
        if len(pos)>3:
        # Set orientation
            self.target.set_orientation(pos[3:7], relative_to)

    def set_joint_mode(self, mode):
        """
            Sets the joint mode for all joints
            @mode - 'position' - position or IK mode
                    'velocity' - force/torque control mode
                    'force' - force/torque control mode
        """
        if mode=="position":
            self.jointmode = 'position'
            for i in range(len(self.joints)):
                self.joints[i].set_joint_mode(JointMode.IK)
        elif mode=='velocity' or mode=='force':
            self.jointmode = 'force'
            for i in range(len(self.joints)):
                self.joints[i].set_joint_mode(JointMode.FORCE)
                self.joints[i].set_motor_enabled(True)
                self.joints[i].set_motor_locked_at_zero_velocity(False)
                self.joints[i].set_control_loop_enabled(False)

    def offset_position(self, pos):
        cpos = self.target.get_position()
        npos = [cpos[0]+pos[0],cpos[1]+pos[1],cpos[2]+pos[2]]
        self.target.set_position(npos)

    def set_joint_position(self, joint_n, pos):
        self.joints[joint_n].set_joint_position(pos)

    def offset_joint_position(self, joint_n, delta):
        pos = self.get_joint_position(joint_n)
        self.joints[joint_n].set_joint_position(pos+delta)

    def set_joint_velocity(self, joint_n, vel):
        assert self.jointmode=='force',\
        'JointMode should be velocity/force \n do limb.set_joint_mode(\'velocity\')'
        self.joints[joint_n].set_joint_target_velocity(vel)

    def get_joint_position(self, joint_n):
        return 	self.joints[joint_n].get_joint_position()

    def get_joint_velocity(self, joint_val):
        return 	self.joints[joint_n].get_joint_velocity()
