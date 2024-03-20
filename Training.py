import numpy as np

import DQN
import Env
import os
import QMIX
import utils
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from QMIX import N_STATES
MEMORY_CAPACITY=2000  #经验回放池大小


def DQN_Training(check_point=False,PATH='./checkpoints/checkpoint_DQN_3agent_3obstacle_8directions_50000.pkl'):
    dqn= DQN.DQNet(MEMORY_CAPACITY,check_point,PATH)
    env=Env.MAPFEnv()

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
                    print(env.state.map,'\n',env.state.obstacle)
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


def QMIX_Training(check_point=False,Path=None):
    #初始化经验回放池
    e_rpm = utils.EpisodeMemory(episode_size=10, num_step=2000)
    rpm = utils.ReplayMemory(e_rpm)
    # ExperienceBuffer={'s':np.zeros((MEMORY_CAPACITY,N_STATES)),'a':np.zeros((MEMORY_CAPACITY,1)),'r':np.zeros((MEMORY_CAPACITY,1)),'s_':np.zeros((MEMORY_CAPACITY,N_STATES))}
    qmix=QMIX.QMIX()
    env = Env.MAPFEnv()
    ExperienceBuffer=[]
    for i_episode in range(400):
        s=env.reset()
        for i_step in range(1000):
            s=s.reshape(1,1,N_STATES)
            action=qmix.choose_action(s,env=env)
            s_, r, done, info = env.step(action)
            if i_step==999:
                done=True
            rpm.append((s, action, r, s_, done),done)   #搜集数据
            s = s_
            if done:
                break

        if len(e_rpm) > MEMORY_CAPACITY:
            qmix.learn(buffer=e_rpm,train_step=i_step)
    return

if __name__== "__main__":
    # DQN_Training(True)
    QMIX_Training()
    #绘图表现视野
    #改choose_action()，不会选择已到终点的agent做动作
    #试试加上unsqueeze训练