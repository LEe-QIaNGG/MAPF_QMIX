import numpy as np
import pickle
import DQN
import Env
import os
import QMIX
import utils
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from utils import NUM_STEP
MEMORY_CAPACITY=20  #经验回放池大小


def DQN_Training(check_point=False,PATH='./checkpoints/checkpoint_DQN_3agent_3obstacle_8directions_7121.pkl'):
    dqn= DQN.DQNet(MEMORY_CAPACITY,check_point,PATH)
    env=Env.MAPFEnv('CTCE')

    print("\nCollecting experience...")
    for i_episode in range(400):
        
        s = env.reset()
        ep_r = 0
        while True:
            env.render()
            if check_point or dqn.memory_counter > MEMORY_CAPACITY:
                a = dqn.choose_action(s,env)
            else:
                a=env.action_space.sample()
            s_, r, done, info = env.step(a)
            
            #保存经验
            dqn.store_transition(s, a, r, s_)
            
            ep_r += r
            #经验回放池被填满，DQN开始学习或更新     
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))

                if dqn.learn_step_counter%10000==0:
                    remaining_dist=env.state.get_remaining_dist()
                    print('Ep: ', i_episode,'step ',dqn.learn_step_counter,'agent_list',
                          env.agent_id,'remaining distance',remaining_dist,' Reward ',ep_r,end='\n')
                    # print(env.state.map,'\n',env.state.obstacle)
                    if remaining_dist>200 or dqn.learn_step_counter>40000:
                        done=True
            if done:
                dqn.save_checkpoint('./checkpoints')
                dqn.learn_step_counter=0
                #游戏结束，退出while循环
                break
            #使用下一个状态来更新当前状态
            s = s_
    env.close()


def QMIX_Training(load_rpm=False,check_point=False,Path=None):
    #初始化经验回放池
    e_rpm = utils.EpisodeMemory(num_episode=MEMORY_CAPACITY)
    rpm = utils.ReplayMemory(e_rpm)
    if load_rpm:
        #读取保存的经验
        with open("./rpm/rpm.pickle", "rb") as file:
            rpm = pickle.load(file)
        with open("./rpm/e_rpm.pickle", "rb") as file:
            e_rpm = pickle.load(file)
    # ExperienceBuffer={'s':np.zeros((MEMORY_CAPACITY,N_STATES)),'a':np.zeros((MEMORY_CAPACITY,1)),'r':np.zeros((MEMORY_CAPACITY,1)),'s_':np.zeros((MEMORY_CAPACITY,N_STATES))}
    qmix=QMIX.QMIX()
    env = Env.MAPFEnv('DTDE')
    for i_episode in range(400):
        print('episode: ',i_episode,end='\n')
        s=env.reset()
        s=env.add_index(s)

        if len(e_rpm) >= MEMORY_CAPACITY:
            #保存经验
                with open("./rpm/rpm.pickle", "wb") as file:
                    pickle.dump(rpm, file)
                with open("./rpm/e_rpm.pickle", "wb") as file:
                    pickle.dump(e_rpm, file)
                qmix.learn(buffer=e_rpm, train_step=i_episode)

        for i_step in range(NUM_STEP):
            # env.render()
            #收集经验
            action = []
            for i in range(env.num_agent):
                a=qmix.choose_action(s,env=env,agent_id=i)
                action.append(a)
            s_, r, done, info = env.step(action)
            s_=env.add_index(s_)
            if i_step==499:
                done=True
            rpm.append((s, action, r, s_, done),done)   #搜集数据

            s = s_
            if done:
                break

    env.close()
    return

if __name__== "__main__":
    # DQN_Training(check_point=True)
    QMIX_Training(load_rpm=True)
    #绘图表现视野
    #改choose_action()，不会选择已到终点的agent做动作
    #试试加上unsqueeze训练