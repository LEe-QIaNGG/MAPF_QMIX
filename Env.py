import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.distance import pdist,squareform
import time
from gym.envs.classic_control import rendering

#奖励的参数
GOAL_REWARD,COLLISION_REWARD,FINISH_REWARD,AWAY_COST,ACTION_COST=2,-3,5,-1,-1
#碰撞的判定距离，智能体的视野距离,智能体的步长
COLLID_E,OBSERVE_DIST,STEP_LEN,GOAL_E=0.5,2.5,0.8,0.2
#状态空间 方向矩阵大小
OBSERVATION_SIZE=4
NUM_OBSTACLE,NUM_AGENTS=3,3 #障碍物个数
MAP_WIDTH=20 #地图为正方形
SCREEN_WIDTH=400
DRAW_MUTIPLE=SCREEN_WIDTH/MAP_WIDTH
NUM_DIRECTIONS=8


class State():
    def __init__(self,num_agent,len_edge):
        self.num_agent=num_agent
        self.num_obstacle=NUM_OBSTACLE
        self.len_edge=len_edge


        #起点，终点，障碍
        self.map,self.obstacle=self.generate_map() #4*num_agent  #num_obstacle*2
        #障碍索引
        self.obstacle_id=np.ones((num_agent,self.num_obstacle))#num_agent*num_obstacle
        #周围的智能体索引，对角线为零
        self.obs_agent=np.ones((num_agent,num_agent))
        np.fill_diagonal(self.obs_agent,0)
        #状态方向矩阵
        self.direction_mat=np.ones((NUM_AGENTS,OBSERVATION_SIZE))*-1
        self.observation=np.concatenate((self.map[:,0:2],self.direction_mat),axis=1)

    def generate_map(self):
        map=np.random.rand(self.num_agent,4)*self.len_edge
        obstacle=np.random.rand(self.num_obstacle,2)*self.len_edge
        # print(self.map)
        return map,obstacle


    def move_agents(self,action):
        #更新位置，若远离goal返回cost，靠近返回0
        
        pos=self.map[action[0],0:2]  #当前智能体的二维坐标
        goal=self.map[action[0],2:4]  #当前智能体终点坐标
        change=STEP_LEN*np.stack([np.cos(action[1]*2*np.pi/NUM_DIRECTIONS),np.sin(action[1]*2*np.pi/NUM_DIRECTIONS)]).reshape(-1,2)
        new_pos=pos+change
        if np.linalg.norm(goal - pos, ord=2) > np.linalg.norm(goal - new_pos, ord=2):
            cost=0
        else:
            cost=AWAY_COST
        self.map[action[0],0:2]=new_pos
        return cost

    
    def observe(self,id):
        #通过简略计算更新id智能体周围智能体和障碍编号，返回状态

        #所有智能体的二维位置
        pos=self.map[:,0:2]

        #得到N*N的曼哈顿距离矩阵
        dist_mat=squareform(pdist(pos,'cityblock'))

        #更新周围智能体索引矩阵
        self.obs_agent=np.where(dist_mat<OBSERVE_DIST,1,0)
        np.fill_diagonal(self.obs_agent, 0)

        #更新障碍索引
        pos_agent_mat=self.obstacle
        x=pos_agent_mat-pos[id]
        # dist_array=pdist(pos_agent_mat-pos[id],'cityblock')
        dist_array=np.linalg.norm(x,ord=2,axis=1)
        self.obstacle_id[id]=np.where(dist_array<OBSERVE_DIST,1,0)
        #计算状态
        dist_array=np.concatenate((dist_mat[id],dist_array),axis=0)  #智能体和障碍的距离向量拼接
        pos_tot=np.concatenate((pos,pos_agent_mat))
        #满足距离的
        index=np.argwhere((dist_array<OBSERVE_DIST) & (dist_array>0))

        if len(index)>0 and len(index)<OBSERVATION_SIZE:
            self.get_direction(pos_tot[index], pos[id], id)
        elif len(index)>OBSERVATION_SIZE:
            pos_tot = pos_tot[index]
            dist_array = dist_array[index]
            index=np.argpartition(dist_array,OBSERVATION_SIZE)[0:OBSERVATION_SIZE]  #得到前OBSERVATION_SIZE大小近的智能体和障碍索引
            #在视野范围外的就不算方向了

            self.get_direction(pos_tot[index],pos[id],id)
            #更新observation
        self.observation = np.concatenate((self.map[:,0:2], self.direction_mat),axis=1)
        return self.observation

    def get_direction(self,near_pos,this_pos,id):
            #根据id号智能体最近的物体位置和自己位置，更新对各物体的方向向量 direction_mat
            difference=near_pos-this_pos
            difference=difference.reshape(-1,2)
            rad=np.arctan(difference[:,1]/difference[:,0])  #-pi/2,pi/2
            x=np.where(difference[:,0]>0,0,np.pi)
            arr=rad+x
            arr=np.pad(arr,(0,OBSERVATION_SIZE-len(arr)),'constant',constant_values=(0,-1))
            self.direction_mat[id]=arr
            return
    
    def eval(self,id):
        #计算id号智能体的reward，返回r和id，若id号智能体到达终点，返回-1
        reward=0

        reward=reward+self.collide(id)
        if self.finish(id):
            reward=reward+GOAL_REWARD
            id=-1
        return reward,id

    def collide(self,id):
        #返回id号智能体碰撞奖励
        #id号智能体当前二维位置
        pos=self.map[id,0:2]

        # 障碍 n*2
        obstacle_mat=self.obstacle.take(np.argwhere(self.obstacle_id[id]==1),axis=0).reshape(-1,2)

        near_agent_index=self.obs_agent[id]
        #id号周围智能体二维位置
        agent_mat=self.map[:,0:2].take(np.argwhere(near_agent_index==1),axis=0).reshape(-1,2)

        #周围智能体和障碍的m*2位置矩阵
        pos_tot=np.concatenate((agent_mat,obstacle_mat),axis=0)

        r=COLLISION_REWARD*self.norm_two(pos_tot-pos,COLLID_E)
        return r
    
    def finish(self,id):
        now=self.map[id,0:2]
        goal=self.map[id,2:4]
        dist=np.linalg.norm(goal-now,ord=2)
        if dist<GOAL_E:
            return True
        else:
            return False
    
    def reset(self):
        self.map,_=self.generate_map()
        self.obstacle_id=np.ones((self.num_agent,self.num_obstacle))
        self.obs_agent = np.ones((self.num_agent, self.num_agent))
        np.fill_diagonal(self.obs_agent, 0)
        self.observation = np.concatenate((self.map[:,0:2], self.direction_mat),axis=1)
        return

    def norm_two(self,mat,e):
        ##小于e即为发生碰撞，返回碰撞个数
        #mat为m*2的矩阵
        dist=np.linalg.norm(mat,ord=2,axis=1)
        return np.sum(dist<e)

    def get_remaining_dist(self):
        #各智能体位置与终点坐标差
        mat=self.map[:,0:2]-self.map[:,2:4]
        return sum(np.linalg.norm(mat,ord=2,axis=1))


    


class MAPFEnv(gym.Env):
    metadata = {'render.modes':['human','rgb_array'],'video.frames_per_second': 2}
    def  __init__(self):
        self.num_agent=NUM_AGENTS
        self.agent_id=list(range(self.num_agent))
        self.len_edge=MAP_WIDTH
        self.viewer = rendering.Viewer(SCREEN_WIDTH*2, SCREEN_WIDTH*2)

        #动作空间简化为离散的N个方向，每次只有一个智能体移动
        # self.action_space=spaces.Box(low=0,high=NUM_DIRECTIONS-1,shape=(1,self.num_agent),dtype=np.int64)
        self.action_space=spaces.Tuple([spaces.Discrete(self.num_agent), spaces.Discrete(NUM_DIRECTIONS)])

        #观察空间为智能体的二维位置，即2*N的nparray
        # self.observation_space=spaces.Discrete(self.num_agent)

        #初始化地图
        self.state=State(self.num_agent,self.len_edge)

    def step(self,action):
        #接受动作，返回状态，奖励，done，info，action为(agent_id,direction)
        if action[0] in self.agent_id:
            away_cost=self.state.move_agents(action)
            #获取奖励，判断是否到终点
            Reward,id=self.state.eval(action[0])
            Reward=Reward+away_cost
            if id==-1:
                #智能体到达终点，活动智能体索引中删除该编号
                self.agent_id.remove(action[0])
                observation=self.state.observation
            else:
                #没有到达终点的话，更新智能体周围智能体和障碍信息
                observation=self.state.observe(action[0])
        else:
            #该智能体已到终点，无需做动作
            observation = self.state.observation
            Reward=ACTION_COST
        
        info={}

        if self.agent_id==[]:
            Reward=Reward+FINISH_REWARD
            done=True
        else:
            done=False

        return observation,Reward,done,info

    def reset(self):
        # self.viewer = rendering.Viewer(SCREEN_WIDTH*3, SCREEN_WIDTH*3)
        self.agent_id=list(range(self.num_agent))
        self.state.reset()
        return self.state.observation
    
    def render(self,mode='human'):
        #画障碍
        for o in self.state.obstacle:
            self.viewer.draw_circle(
                COLLID_E*DRAW_MUTIPLE/2, 4, True, color=(0, 0, 0)
            ).add_attr(rendering.Transform(o*DRAW_MUTIPLE+[SCREEN_WIDTH/2,SCREEN_WIDTH/2]))

        #画智能体
        for a in self.state.map[:,0:2]:
            self.viewer.draw_circle(
                COLLID_E*DRAW_MUTIPLE/2, 30, True, color=(0, 0, 255)
            ).add_attr(rendering.Transform(a*DRAW_MUTIPLE+[SCREEN_WIDTH/2,SCREEN_WIDTH/2]))

        #画终点
        for g in self.state.map[:,2:4]:
            self.viewer.draw_circle(
                COLLID_E*DRAW_MUTIPLE/2, 6, True, color=(0, 255, 0)
            ).add_attr(rendering.Transform(g*DRAW_MUTIPLE+[SCREEN_WIDTH/2,SCREEN_WIDTH/2]))
        return self.viewer.render(return_rgb_array=mode == 'human')
    
    def seed(self,seed=None):
        pass
    
    def close(self):
        self.viewer.close()


if __name__== "__main__":
    env=MAPFEnv()
    # print(env.action_space.sample())
    # # check_env(env)
    #测试
    for epoch in range(2):
        for epoch in range(5):
            env.reset()
            print('Epoch', epoch+1, ': ',end='\n')
            # print('remaining distance:',env.state.get_remaining_dist(), end='')
            for i in range(5):
                _,r,_,_=env.step(env.action_space.sample())     # 随机选择一个动作执行
                print('remaining distance:', env.state.get_remaining_dist(),' Reward:',r, end='\n')
                env.render()    # 刷新画面
                time.sleep(0.2)

    env.close()
