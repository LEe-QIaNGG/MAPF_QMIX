import DQN
import Env

MEMORY_CAPACITY = 2000  #经验回放池大小

def DQN_Training():
    dqn= DQN.DQNet(MEMORY_CAPACITY)
    env=Env.MAPFEnv()

    print("\nCollecting experience...")
    for i_episode in range(400):
        
        s = env.reset()
        ep_r = 0
        while True:
            env.render()
            
            a = dqn.choose_action(s,env)
            s_, r, done, info = env.step(a)
            
            # # modify the reward based on the environment state
            # x, x_dot, theta, theta_dot = s_
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # r = r1 + r2
            
            #保存经验
            dqn.store_transition(s, a, r, s_)
            
            ep_r += r
            #如果经验回放池被填满，DQN开始学习或更新     
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))
            
            if done:
                #游戏结束，退出while循环
                break
            #使用下一个状态来更新当前状态
            s = s_  

if __name__== "__main__":
    DQN_Training()