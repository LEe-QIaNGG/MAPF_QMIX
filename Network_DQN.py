import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import MAPFenv

BATCH_SIZE = 32     # batch size of sampling process from buffer
LR = 0.01           # learning rate
EPSILON = 0.9       # epsilon used for epsilon greedy approach
GAMMA = 0.9         # discount factor
TARGET_NETWORK_REPLACE_FREQ = 100       # How frequently target netowrk updates
MEMORY_CAPACITY = 2000                  # The capacity of experience replay buffer

env=MAPFenv(3,10)
N_ACTIONS = env.action_space.n  # 2 actions
N_STATES = env.observation_space.shape[0] # 4 states


# Define the network used in both target net and the net for training
class Net(nn.Module):
    def __init__(self):
        # Define the network structure, a very simple fully connected network
        super(Net, self).__init__()
        # Define the structure of fully connected network
        self.fc1 = nn.Linear(N_STATES, 10)  # layer 1
        self.fc1.weight.data.normal_(0, 0.1) # in-place initilization of weights of fc1
        self.out = nn.Linear(10, N_ACTIONS) # layer 2
        self.out.weight.data.normal_(0, 0.1) # in-place initilization of weights of fc2
        
        
    def forward(self, x):
        # Define how the input data pass inside the network
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
        



# Define the DQN network and its corresponding methods
class DQN(object):
    def __init__(self):
        # -----------Define 2 networks (target and training)------#
        self.eval_net, self.target_net = Net(), Net()
        # Define counter, memory size and loss function
        self.learn_step_counter = 0 # count the steps of learning process
        self.memory_counter = 0 # counter used for experience replay buffer
        
        # ----Define the memory (or the buffer), allocate some space to it. The number of columns depends on 4 elements, s, a, r, s_, the total is N_STATES*2 + 2---#
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2)) 
        
        #------- Define the optimizer------#
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        
        # ------Define the loss function-----#
        self.loss_func = nn.MSELoss()

    def  choose_action(self, x):
        # This function is used to make decision based upon epsilon greedy
        
        x = torch.unsqueeze(torch.FloatTensor(x), 0) # add 1 dimension to input state x
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            # use epsilon-greedy approach to take action
            actions_value = self.eval_net.forward(x)
            #print(torch.max(actions_value, 1)) 
            # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
            # what we need is the index in this function, representing the action of cart.
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            # action = np.random.randint(0, N_ACTIONS)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            action=env.action_space.sample()
        return action
    
        
    def store_transition(self, s, a, r, s_):
        # This function acts as experience replay buffer        
        transition = np.hstack((s, [a, r], s_)) # horizontally stack these vectors
        # if the capacity is full, then use index to replace the old memory with new one
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        
    
    def learn(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network, and how to implement
        # backward propagation.
        
        # update the target network every fixed steps
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        # Determine the index of Sampled batch from buffer
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE) # randomly select some data from buffer
        # extract experiences of batch size from buffer.
        b_memory = self.memory[sample_index, :]
        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        # convert long int type to tensor
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))
        
        # calculate the Q value of state-action pair
        q_eval = self.eval_net(b_s).gather(1, b_a) # (batch_size, 1)
        #print(q_eval)
        # calculate the q value of next state
        q_next = self.target_net(b_s_).detach() # detach from computational graph, don't back propagate
        # select the maximum q value
        #print(q_next)
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) # (batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad() # reset the gradient to zero
        loss.backward()
        self.optimizer.step() # execute back propagation for one step