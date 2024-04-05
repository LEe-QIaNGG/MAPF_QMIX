import Env
import DQN
import QMIX
from statistics import mean
from Env import STEP_LEN
MAX_STEP=3000
NUM_EPISODE=50

def evaluate(mode):
    assert mode=='QMIX' or mode=='DQN'
    if mode=='QMIX':
        qmix=QMIX.QMIX(load_checkpoint=True,path='./checkpoints/checkpoint_QMIX_3agent_3obstacle_8directions.pkl')
        env = Env.MAPFEnv('DTDE',render=True)
    else:
        dqn = DQN.DQNet(200, load_checkpoint=True, PATH='./checkpoints/checkpoint_DQN_3agent_3obstacle_8directions.pkl')
        env = Env.MAPFEnv('CTCE',render=True)

    # 总指标
    shortcut_rate = []  # 实际走过路径和直线距离的比值
    Reward = []
    num_fail = 0  # 统计未完成游戏的episode数

    for i_episode in range(NUM_EPISODE):
        s=env.reset()
        if mode=='QMIX':
            s=env.add_index(s)

        # 统计局中变量
        total_dist=env.state.get_remaining_dist()     #获取每局游戏刚开始时总路程
        done=False
        i_step=0
        fail=False    #是否完成
        dist_travelled=0   #统计实际路程
        R=0                #统计奖励


        while done!=True:
            env.render()
            action = []
            if mode=='QMIX':
                for i in range(env.num_agent):
                    a=qmix.choose_action(s,env=env,agent_id=i,epsilon=0.8)
                    action.append(a)
                dist_travelled = dist_travelled + len(env.agent_id) * STEP_LEN
            else:
                action = dqn.choose_action(s, env, epsilon=1)
                dist_travelled=dist_travelled+STEP_LEN

            s_, r, done, info = env.step(action)
            R=R+r
            # print('info；',info)
            if mode=='QMIX':
                s_=env.add_index(s_)
            if i_step>MAX_STEP:
                fail=True
                num_fail=num_fail+1
                done=True
            s = s_
            i_step=i_step+1


        shortcut_rate.append(dist_travelled/total_dist)
        Reward.append(R)
    print('测试{}局，完成{}局，实际走过路径和直线距离的比值为{}，其均值为{},平均奖励为{}'.format(NUM_EPISODE,NUM_EPISODE-num_fail,shortcut_rate,sum(shortcut_rate)/NUM_EPISODE,sum(Reward)/NUM_EPISODE))
    print()

#安全性指标

if __name__== "__main__":
    evaluate('DQN')