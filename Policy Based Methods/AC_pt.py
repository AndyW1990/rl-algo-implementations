# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 21:20:00 2020

@author: andrew.whitworth
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym


class Network(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, output_dim):
        super(Network, self).__init__()
        
        self.lr = lr        
        self.input_dims = input_dims
        self.fc1_dims = fc2_dims
        self.fc2_dims = fc2_dims
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')        
        self.to(self.device)
        
        
    def forward(self, obs):
        state = T.Tensor(obs).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    
class Agent(object):
    def __init__(self, input_dims, n_actions, lr=0.001, gamma=0.99, 
                 save_dir='C:\\ANDY\\Neural Networks\\RL\\Learning'):
        
        self.gamma = gamma
        self.action_memory = []        
        self.reward_memory = []
        self.value_memory = []        
        self.policy = Network(lr/10, input_dims, 2028, 512, n_actions)
        self.value = Network(lr, input_dims, 256, 256, 1)
        self.model_loc = save_dir
        
    def choose_action(self, obs):
        probs = F.softmax(self.policy.forward(obs))
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        
        return action.item()
    
    def store_transitions(self, state, reward):
        self.value_memory.append(self.value.forward(state))
        self.reward_memory.append(reward)
    


    def learn(self):
        self.policy.optimizer.zero_grad()
        self.value.optimizer.zero_grad()        
        
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        G = T.tensor(G, dtype=T.float).to(self.policy.device)
         
        pi_loss = 0
        v_loss = 0        
        for g, state_value, logprob in zip(G, self.value_memory, self.action_memory):
            pi_loss -= logprob * (g - state_value.item())
            v_loss += ((state_value - g)**2)
        
        loss = (pi_loss + v_loss)
        loss.backward()
        self.policy.optimizer.step()
        self.value.optimizer.step()
        
        self.action_memory = []        
        self.reward_memory = []       
        self.value_memory = []
        
    def save_model(self):
        T.save(self.policy, self.model_loc + '\\policy_model.pt')
        T.save(self.value, self.model_loc + '\\value_model.pt')

        
   
if __name__ == '__main__':
    
    env = gym.make('CartPole-v1')
    agent = Agent(input_dims=env.observation_space.shape, 
                  n_actions=env.action_space.n, lr=0.001, gamma=0.99)
    
    n_episodes = 200
    score_history = []
    
    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
        
        while not done:
            #env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transitions(observation, reward)
            observation = observation_            
            score += reward
        score_history.append(score)
        
        agent.learn()            
        
        avg_score = np.mean(score_history[-100:])        
        print('episode ', i, 
              'score %.1f' % score,
              'avg_score %.1f' % avg_score)
        
    agent.save_model()


#        pi_losses = T.stack(pi_losses)
#        v_losses = T.stack(v_losses)        
#        
##        mean_pi = pi_losses.mean()
##        std_pi = np.std(pi_losses.tolist()) if np.std(pi_losses.tolist()) > 0 else 1
##        pi_losses = (pi_losses-mean_pi)/std_pi
##        
#
##        mean_v = v_losses.mean()
##        std_v = np.std(v_losses.tolist()) if np.std(v_losses.tolist()) > 0 else 1
##        v_losses = (v_losses-mean_v)/std_v
#        
#        loss = pi_losses.sum() - v_losses.sum()    
    
    