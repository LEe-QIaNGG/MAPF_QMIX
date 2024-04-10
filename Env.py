import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.distance import pdist,squareform
import time
from gym.envs.classic_control import rendering
import argparse

#奖励的参数
GOAL_REWARD,COLLISION_REWARD,FINISH_REWARD,AWAY_COST,ACTION_COST=2,-3,5,-4,-0.5
#碰撞的判定距离，智能体的视野距离,智能体的步长
COLLID_E,OBSERVE_DIST,STEP_LEN,GOAL_E=0.5,2.5,0.1,0.2
#状态空间 方向矩阵大小
OBSERVATION_SIZE=4
NUM_OBSTACLE,NUM_AGENTS=8,4 #障碍物个数
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
        self.direction_mat=np.zeros((NUM_AGENTS,OBSERVATION_SIZE*2))
        self.observation=np.concatenate((self.map,self.direction_mat),axis=1)

    def generate_map(self):
        #防止生成的终点太近
        goal_collide=True
        while goal_collide:
            map=np.random.rand(self.num_agent,4)*self.len_edge
            dist=pdist(map[:,2:4])
            if min(dist)>3:
                goal_collide=False

        #防止生成的障碍和终点太近
        obs_collide=True
        while obs_collide:
            min_dist=[]
            obstacle=np.random.rand(self.num_obstacle,2)*self.len_edge
            for point in map[:,2:4]:
                mat=abs(point-obstacle)
                min_dist.append(min(np.sum(mat,axis=1)))
            if min(min_dist)>2:
                obs_collide=False
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
            observe_array=np.pad(pos_tot[index].reshape(-1),(0,(OBSERVATION_SIZE-len(index))*2),'constant',constant_values=(0,0))
            self.direction_mat[id]=observe_array
        elif len(index)>OBSERVATION_SIZE:
            index=np.argpartition(dist_array,OBSERVATION_SIZE)[0:OBSERVATION_SIZE]  #得到前OBSERVATION_SIZE大小近的智能体和障碍索引
            #在视野范围外的就不算方向了

            self.direction_mat[id]=pos_tot[index].reshape(-1)
            #更新observation
        else:
            self.direction_mat[id]= np.zeros(OBSERVATION_SIZE*2)
        self.observation = np.concatenate((self.map,self.direction_mat),axis=1)
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
        self.direction_mat=np.zeros((NUM_AGENTS,OBSERVATION_SIZE*2))
        self.observation = np.concatenate((self.map, self.direction_mat),axis=1)
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
    def  __init__(self,mode,render=False):
        #mode值：CTCE或DTDE
        self.num_agent=NUM_AGENTS
        self.agent_id=list(range(self.num_agent))
        self.len_edge=MAP_WIDTH
        assert mode=='DTDE' or mode=='CTCE'
        self.mode=mode
        self.flag_render=render


        #动作空间简化为离散的N个方向，每次只有一个智能体移动
        # self.action_space=spaces.Box(low=0,high=NUM_DIRECTIONS-1,shape=(1,self.num_agent),dtype=np.int64)
        if mode=='CTCE':
            self.action_space = spaces.Discrete(self.num_agent * NUM_DIRECTIONS)
        elif mode=='DTDE':
            #在step前将拼接好的action传进去
            self.action_space = spaces.Discrete(NUM_DIRECTIONS)

        #观察空间为智能体的二维位置，即2*N的nparray
        # self.observation_space=spaces.Discrete(self.num_agent)
        #初始化地图
        self.state=State(self.num_agent,self.len_edge)
        if render:
            #画布
            self.viewer = rendering.Viewer(SCREEN_WIDTH*2, SCREEN_WIDTH*2)

    def step(self,index):
        if self.mode=='CTCE':   #这种情况参数为空间为num_agent * NUM_DIRECTIONS的一个数值
            #返回的是一个agent做动作后的s，r，done，info
            action=tuple([int(index/NUM_DIRECTIONS),int(index%NUM_DIRECTIONS)])
            observation,Reward,done,info=self.step_single_agent(action)
            return observation, Reward, done, info

        elif self.mode=='DTDE':   #这种情况参数为shape n*1的向量，第二位空间为n direction

            n=len(index)
            observation,Reward,Info=[],0,[]
            for id,direction in enumerate(index):
                action=[id,direction]
                o,r,done,info=self.step_single_agent(action)
                observation=np.concatenate((observation,o[id]))
                Reward=Reward+r
                Info.append(info)
                if done:
                    if id!=n-1:
                        # 填充至形状一样
                        observation=np.concatenate((observation,np.zeros((NUM_AGENTS-id-1)*(4+2*OBSERVATION_SIZE))))
                    break

            return observation.reshape(NUM_AGENTS,-1),Reward,done,Info


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

    def step_single_agent(self,action):
        #action: (id,direction)
        info = []
        # 接受动作，返回状态，奖励，done，info，action为(agent_id,direction)
        if action[0] in self.agent_id:
            away_cost = self.state.move_agents(action)
            # 获取奖励，判断是否到终点
            Reward, id = self.state.eval(action[0])
            Reward = Reward + away_cost
            if id == -1:
                # 智能体到达终点，活动智能体索引中删除该编号
                self.agent_id.remove(action[0])
                self.state.direction_mat[action[0]] = np.ones(OBSERVATION_SIZE * 2) * -1
                observation = np.concatenate((self.state.map, self.state.direction_mat), axis=1)
                info = [action[0], '智能体到达终点']
            else:
                info = [action[0], '智能体移动', 'away_cost:', away_cost]
                # 没有到达终点的话，更新智能体周围智能体和障碍信息
                observation = self.state.observe(action[0])
        else:
            # 该智能体已到终点，无需做动作
            info = '智能体已在终点'
            observation = self.state.observation
            Reward = ACTION_COST

        if self.agent_id == []:
            Reward = Reward + FINISH_REWARD
            done = True
        else:
            done = False

        return observation, Reward, done, info

    def add_index(self,state):
        #给DTDE的状态加id index
        index=np.array(range(NUM_AGENTS)).reshape(NUM_AGENTS,-1)
        state=np.concatenate((index,state),axis=1)
        return state

def qmix_args(args):
    args.rnn_hidden_dim = 512
    args.two_hyper_layers = True
    args.qmix_hidden_dim = 128
    args.hyper_hidden_dim=256
    args.lr = 5e-6

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.1
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch
    args.n_episodes = 10

    # the number of the train steps in one epoch
    args.train_steps = 1

    # # how often to evaluate
    args.evaluate_cycle = 25

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 20

    # how often to update the target_net
    args.target_update_cycle = 20

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


def get_common_args():
    parser = argparse.ArgumentParser()

    # the environment setting
    parser.add_argument('--obs_space', type=int, default=12, help='observation space')
    parser.add_argument('--state_space', type=int, default=NUM_AGENTS*12, help='observation space')

    parser.add_argument('--action_space', type=int, default=8, help='action space')
    parser.add_argument('--num_actions', type=int, default=8, help='number of agents')
    parser.add_argument('--num_agents', type=int, default=NUM_AGENTS, help='number of agents')
    parser.add_argument('--max_episode_steps', type=int, default=800, help='number of agents')

    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='the directory of save the replay')

    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="ADAM", help='optimizer')
    parser.add_argument('--n_evaluate_episode', type=int, default=5, help='number of the episode to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--threshold', type=float, default=19.9, help='threshold to judge whether win')
    args = parser.parse_args()
    return args



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
