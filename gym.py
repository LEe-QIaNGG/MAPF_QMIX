import gym
import numpy as np
from gym import spaces

class MAPFEnv(gym.env):
    def  __init__(self,num_agent):
        self.N=num_agent
        #动作空间简化为离散的36个方向，加上静止动作
        self.action_space=spaces.Discrete(37)
        #观察空间为智能体的二维位置，即2*N的nparray
        self.observation_space=spaces.Box(low=0,high=100, shape= (2,self.N) , dtype=np.float32)