import pickle
import DQN
import Env
import os
from MyQmix import QMIX, utils
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from Env import STEP_LEN

NUM_STEP=3000
MEMORY_CAPACITY=200  #经验回放池大小
EPSILON=0.8       #epsilon greedy方法
NUM_EPISODE=3000


def DQN_Training(check_point=False,render=False,PATH='./checkpoints/checkpoint_DQN_2agent_2obstacle_8directions.pkl'):
    dqn= DQN.DQNet(200,check_point,PATH)
    env=Env.MAPFEnv('DTDE',render=render)

    #测试总指标
    shortcut_rate = []  # 实际走过路径和直线距离的比值
    num_fail = 0  # 统计未完成游戏的episode数
    Reward = []

    print("\nCollecting experience...")
    for i_episode in range(NUM_EPISODE):
        # 统计局中变量
        total_dist=env.state.get_remaining_dist()     #获取每局游戏刚开始时总路程
        dist_travelled=0   #统计实际路程

        dqn.episode = dqn.episode + 1
        if i_episode%300==0:
            s = env.reset(change_map=True)
        else:
            s = env.reset(change_map=False)
        ep_r = 0
        while True:
            if render:
                env.render()
            if check_point or dqn.memory_counter > MEMORY_CAPACITY:
                action = []
                for i in range(env.num_agent):
                    a = dqn.choose_action(s[i],env,EPSILON)
                    action.append(a)
            else:
                action = []
                for i in range(env.num_agent):
                    a= env.action_space.sample()
                    action.append(a)


            s_, r, done, info = env.step(action)
            r=r-dqn.learn_step_counter/200
            if info!='智能体已在终点':
                dist_travelled = dist_travelled + STEP_LEN
            
            #保存经验
            dqn.store_transition(s, action, r, s_)
            
            ep_r += r
            #经验回放池被填满，DQN开始学习或更新     
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', dqn.episode, ' |', 'Ep_r: ', round(ep_r, 2))

                if dqn.learn_step_counter%2000==0:
                    remaining_dist=env.state.get_remaining_dist()
                    print('Ep: ', dqn.episode,'step ',dqn.learn_step_counter,'agent_list',
                          env.agent_id,'remaining distance',remaining_dist,' Reward ',ep_r,end='\n')
                    # print(env.state.map,'\n',env.state.obstacle)
                    if remaining_dist>100 or dqn.learn_step_counter>800:
                        num_fail = num_fail + 1
                        print('超出步数限制')
                        done=True
            if done:
                if i_episode%5==0:
                    dqn.save_checkpoint('./checkpoints')
                dqn.learn_step_counter=0
                #游戏结束，退出while循环
                break
            #使用下一个状态来更新当前状态
            s = s_
        if render:
            plt.clf()  # 清除上一幅图像
            plt.xlabel('episode', fontdict={"family": "Times New Roman", "size": 15})
            plt.ylabel('loss', fontdict={"family": "Times New Roman", "size": 15})
            plt.plot(dqn.loss)
            plt.pause(0.01)  # 暂停0.01秒
            plt.ioff()  # 关闭画图的窗口

        shortcut_rate.append(dist_travelled / total_dist)
        Reward.append(ep_r)

    print('测试{}局，完成{}局，实际走过路径和直线距离的比值均值为{},平均奖励为{}'.format(NUM_EPISODE, NUM_EPISODE - num_fail,
                                                                sum(shortcut_rate) / NUM_EPISODE,
                                                                sum(Reward) / NUM_EPISODE))
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
    qmix= QMIX.QMIX(check_point, path=Path)
    env = Env.MAPFEnv('DTDE',render=render)

    #测试总指标
    shortcut_rate = []  # 实际走过路径和直线距离的比值
    num_fail = 0  # 统计未完成游戏的episode数
    Reward = []

    for i_episode in range(NUM_EPISODE):
        # 统计局中变量
        total_dist=env.state.get_remaining_dist()     #获取每局游戏刚开始时总路程
        dist_travelled=0   #统计实际路程
        R=0

        qmix.episode=qmix.episode+1
        print('episode: ',qmix.episode,end='\n')
        s=env.reset()
        qmix.init_hidden()

        #保存模型
        if i_episode%10==0 and i_episode!=0:
            qmix.save_checkpoint('./checkpoints')

        if i_episode%5==0:
            #保存经验
            with open("./rpm/rpm.pickle", "wb") as file:
                pickle.dump(rpm, file)
            with open("./rpm/e_rpm.pickle", "wb") as file:
                pickle.dump(e_rpm, file)

        for i_step in range(NUM_STEP):
            if len(e_rpm.buffer) >= MEMORY_CAPACITY and i_step%1000==0:
                qmix.learn(buffer=e_rpm, train_step=i_episode*3+i_step/1000)

            if render:
                env.render()
            #收集经验
            action = []
            for i in range(env.num_agent):
                a=qmix.choose_action(s,env=env,agent_id=i,epsilon=EPSILON)
                action.append(a)
            s_, r, done, info = env.step(action)

            #统计路程
            novalid=info.count('智能体已在终点')
            dist_travelled = dist_travelled + (env.num_agent-novalid) * STEP_LEN
            R=R+r

            if i_step==NUM_STEP-1:
                done=True
            rpm.append((s, action, r, s_, done,0.),done)   #搜集数据   添加的最后一项为padded，标志是否为一局中在规定步数内走完的数据
            s = s_

            if done:
                if done and i_step!=NUM_STEP-1:
                    print('走完一局')
                else:
                    num_fail = num_fail + 1
                    print('超过步数限制')
                break

        shortcut_rate.append(dist_travelled/total_dist)
        Reward.append(R)

        if render:
            plt.clf()  # 清除上一幅图像
            plt.xlabel('episode', fontdict={"family": "Times New Roman", "size": 15})
            plt.ylabel('loss', fontdict={"family": "Times New Roman", "size": 15})
            plt.plot(qmix.loss)
            plt.pause(0.01)  # 暂停0.01秒
            plt.ioff()  # 关闭画图的窗口

    print('测试{}局，完成{}局，实际走过路径和直线距离的比值均值为{},平均奖励为{}'.format(NUM_EPISODE, NUM_EPISODE - num_fail,
                                                                sum(shortcut_rate) / NUM_EPISODE,
                                                                sum(Reward) / NUM_EPISODE))

    if render:
        env.close()

    return

if __name__== "__main__":
    DQN_Training(check_point=True,render=True)
    # QMIX_Training(load_rpm=True,render=True,check_point=True )
    #绘图表现视野
    #改choose_action()，不会选择已到终点的agent做动作
