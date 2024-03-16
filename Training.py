import DQN
import Env
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

MEMORY_CAPACITY=8000  #经验回放池大小


def DQN_Training(check_point=False,PATH='./checkpoints/checkpoint_DQN_3agent_3obstacle_8directions_8763.pkl'):
    dqn= DQN.DQNet(MEMORY_CAPACITY,check_point,PATH)
    env=Env.MAPFEnv()

    print("\nCollecting experience...")
    for i_episode in range(400):
        
        s = env.reset()
        ep_r = 0
        while True:
            env.render()

            a = dqn.choose_action(s,env)
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
                    if remaining_dist>10000 or dqn.learn_step_counter>200000:
                        done=True
            if done:
                dqn.save_checkpoint('./checkpoints')
                dqn.learn_step_counter=0
                env.close()
                #游戏结束，退出while循环
                break
            #使用下一个状态来更新当前状态
            s = s_
    env.close()

if __name__== "__main__":
    DQN_Training(True)
    #精简状态空间表示，绘图表现视野