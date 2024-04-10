import numpy as np
from gym.envs.classic_control import rendering
import torch
from torch.distributions import one_hot_categorical
import time
SCREEN_WIDTH=400

class RolloutWorker:
        def __init__(self, env, agents, args):
                self.env = env
                self.agents = agents
                # self.episode_limit = args.episode_limit
                self.num_actions = args.num_actions
                self.num_agents = args.num_agents
                self.state_space = args.state_space
                self.obs_space = args.obs_space
                self.args = args

                self.epsilon = args.epsilon
                self.anneal_epsilon = args.anneal_epsilon
                self.min_epsilon = args.min_epsilon

                print('init rollout worker')

        def generate_episode(self, episode_num=None, evaluate=False):
                if evaluate:
                        self.env.viewer = rendering.Viewer(SCREEN_WIDTH*2, SCREEN_WIDTH*2)
                o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
                self.env.reset()
                terminated = False
                fail=False

                step = 0
                episode_reward = 0

                last_action = np.zeros((self.args.num_agents, self.args.num_actions))
                self.agents.policy.init_hidden(1)

                # epsilon
                epsilon = 0 if evaluate else self.epsilon
                if self.args.epsilon_anneal_scale == 'episode':
                        epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
                if self.args.epsilon_anneal_scale == 'epoch':
                        if episode_num == 0:
                                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

                while not terminated:
                        if evaluate:
                                self.env.render()
                        # time.sleep(0.2)
                        # obs = self.env.get_obs()
                        # state = self.env.get_state()
                        obs = self.env.state.observation
                        state=obs.reshape(-1)
                        """
                        State and Observations

                        """
                        actions, avail_actions, actions_onehot = [], [], []
                        for agent_id in range(self.num_agents):
                                # avail_action = self.env.get_avail_agent_actions(agent_id)
                                avail_action = None
                                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                                   avail_action, epsilon, evaluate)
                                action_onehot = np.zeros(self.args.num_actions)
                                action_onehot[action] = 1
                                actions.append(action)
                                actions_onehot.append(action_onehot)
                                # avail_actions.append(avail_action)
                                last_action[agent_id] = action_onehot

                        _,reward, terminated, info = self.env.step(actions)
                        if step == self.args.max_episode_steps - 1:
                                terminated = 1
                                fail=True

                        o.append(obs)
                        s.append(state)
                        u.append(np.reshape(actions, [self.num_agents, 1]))
                        u_onehot.append(actions_onehot)
                        # avail_u.append(avail_actions)
                        r.append([reward])
                        terminate.append([terminated])
                        padded.append([0.])
                        episode_reward += reward
                        step += 1
                        # if terminated:
                        #     time.sleep(1)
                        if self.args.epsilon_anneal_scale == 'step':
                                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
                o.append(obs)
                s.append(state)
                o_next = o[1:]
                s_next = s[1:]
                o = o[:-1]
                s = s[:-1]

                # 最后一个obs需要单独计算一下avail_action，到时候需要计算target_q
                # avail_actions = []
                # for agent_id in range(self.num_agents):
                # avail_action = self.env.get_avail_agent_actions(agent_id)
                # avail_actions.append(avail_action)
                # avail_u.append(avail_actions)
                # avail_u_next = avail_u[1:]
                # avail_u = avail_u[:-1]

                for i in range(step, self.args.max_episode_steps):  # 没有的字段用0填充，并且padded为1
                        o.append(np.zeros((self.num_agents, self.obs_space)))
                        u.append(np.zeros([self.num_agents, 1]))
                        s.append(np.zeros(self.state_space))
                        r.append([0.])
                        o_next.append(np.zeros((self.num_agents, self.obs_space)))
                        s_next.append(np.zeros(self.state_space))
                        u_onehot.append(np.zeros((self.num_agents, self.num_actions)))
                        # avail_u.append(np.zeros((self.n_agents, self.n_actions)))
                        # avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
                        padded.append([1.])
                        terminate.append([1.])

                '''
                (o[n], u[n], r[n], o_next[n], avail_u[n], u_onehot[n])组成第n条经验，各项维度都为(episode数，transition数，n_agents, 自己的具体维度)
                 因为avail_u表示当前经验的obs可执行的动作，但是计算target_q的时候，需要obs_net及其可执行动作，
                '''
                episode = dict(o=o.copy(),
                               s=s.copy(),
                               u=u.copy(),
                               r=r.copy(),
                               avail_u=avail_u.copy(),
                               o_next=o_next.copy(),
                               s_next=s_next.copy(),
                               # avail_u_next=avail_u_next.copy(),
                               u_onehot=u_onehot.copy(),
                               padded=padded.copy(),
                               terminated=terminate.copy()
                               )

                for key in episode.keys():
                        episode[key] = np.array([episode[key]])
                if not evaluate:
                        self.epsilon = epsilon
                if evaluate:
                        self.env.close()

                return episode, episode_reward,fail