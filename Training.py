import numpy as np
import pickle
import DQN
import Env
import os
import QMIX
import utils
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from utils import NUM_STEP
MEMORY_CAPACITY=80  #经验回放池大小
EPSILON=0.9       #epsilon greedy方法


def DQN_Training(check_point=False,render=False,PATH='./checkpoints/checkpoint_DQN_3agent_3obstacle_8directions.pkl'):
    dqn= DQN.DQNet(200,check_point,PATH)
    env=Env.MAPFEnv('CTCE',render=render)

    print("\nCollecting experience...")
    for i_episode in range(400):
        dqn.episode = dqn.episode + 1
        s = env.reset()
        ep_r = 0
        while True:
            if render:
                env.render()
            if check_point or dqn.memory_counter > MEMORY_CAPACITY:
                a = dqn.choose_action(s,env,EPSILON)
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
                    print('Ep: ', dqn.episode, ' |', 'Ep_r: ', round(ep_r, 2))

                if dqn.learn_step_counter%10000==0:
                    remaining_dist=env.state.get_remaining_dist()
                    print('Ep: ', dqn.episode,'step ',dqn.learn_step_counter,'agent_list',
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

    if render:
        env.close()


def QMIX_Training(load_rpm=False,render=False,check_point=False,Path='./checkpoints/checkpoint_QMIX_3agent_3obstacle_8directions.pkl'):
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
    qmix=QMIX.QMIX(check_point,path=Path)
    env = Env.MAPFEnv('DTDE',render=render)
    for i_episode in range(400):
        qmix.episode=qmix.episode+1
        print('episode: ',i_episode,end='\n')
        s=env.reset()
        s=env.add_index(s)

        if i_episode%10==0 and i_episode!=0:
            qmix.save_checkpoint('./checkpoints')

        if i_episode%5==0:
            #保存经验
            with open("./rpm/rpm.pickle", "wb") as file:
                pickle.dump(rpm, file)
            with open("./rpm/e_rpm.pickle", "wb") as file:
                pickle.dump(e_rpm, file)

        if len(e_rpm.buffer) >= MEMORY_CAPACITY:
            qmix.learn(buffer=e_rpm, train_step=i_episode)

        for i_step in range(NUM_STEP):
            if render:
                env.render()
            #收集经验
            action = []
            for i in range(env.num_agent):
                a=qmix.choose_action(s,env=env,agent_id=i,epsilon=EPSILON)
                action.append(a)
            s_, r, done, info = env.step(action)
            # print('info；',info)
            s_=env.add_index(s_)
            if i_step==NUM_STEP-1:
                done=True
            rpm.append((s, action, r, s_, done,0.),done)   #搜集数据   添加的最后一项为padded，标志是否为一局中在规定步数内走完的数据
            s = s_

            if done:
                if done and i_step!=NUM_STEP-1:
                    print('走完一局')
                else:
                    print('超过步数限制')
                break
        if render:
            plt.clf()  # 清除上一幅图像
            plt.xlabel('episode', fontdict={"family": "Times New Roman", "size": 15})
            plt.ylabel('loss', fontdict={"family": "Times New Roman", "size": 15})
            plt.plot(qmix.loss)
            plt.pause(0.01)  # 暂停0.01秒
            plt.ioff()  # 关闭画图的窗口
    if render:
        env.close()

    return

if __name__== "__main__":
    # DQN_Training(check_point=True,render=True)
    QMIX_Training(load_rpm=True,render=False,check_point=True )
    #绘图表现视野
    #改choose_action()，不会选择已到终点的agent做动作
    #试试加上unsqueeze训练