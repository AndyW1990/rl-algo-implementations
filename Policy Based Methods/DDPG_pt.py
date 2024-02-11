# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 09:58:33 2021

@author: andrew.whitworth
"""

import gym
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done        
        
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]     
        
        return states, actions, rewards, new_states, dones
    
        
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=512, fc2_dims=512,
                 chkpt_dir='DDPG_files/', name='actor_torch'):
        super(ActorNetwork, self).__init__()
        
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Tanh())
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')      
        self.to(self.device)
        
        
    def forward(self, state):
        mu = self.actor(state)
        
        return mu   
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))    
        
        
class CriticNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=512, fc2_dims=512,
             chkpt_dir='DDPG_files/', name='critic_torch'):
        super(CriticNetwork, self).__init__()
        
        dims = [i + x for i, x in zip(input_dims, (n_actions,))][0]
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.critic = nn.Sequential(
                nn.Linear(dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')        
        self.to(self.device)
        
    def forward(self, state, action):
        value = self.critic(T.cat((state, action),1))
        
        return value       
       
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))    
                
               
class Agent():
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
                 gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 fc1=400, fc2=300, batch_size=64, noise=0.1):        
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        
        self.actor = ActorNetwork(n_actions=n_actions, input_dims=input_dims, 
                                  alpha=alpha)
        self.critic = CriticNetwork(n_actions=n_actions, input_dims=input_dims, 
                                    alpha=beta)            

        self.target_actor = ActorNetwork(n_actions=n_actions, 
                                         input_dims=input_dims, alpha=alpha,
                                         name='target_actor_torch')
        self.target_critic = CriticNetwork(n_actions=n_actions, 
                                           input_dims=input_dims, alpha=beta, 
                                           name='target_critic_torch')

        self.update_network_parameters(tau=1)
        
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        dict_targets = dict(self.target_actor.named_parameters())
        
        for name, weight in self.actor.named_parameters():
            if name in dict_targets:
                dict_targets[name].data.copy_(weight.data*tau + dict_targets[name].data*(1-tau))
        self.target_actor.load_state_dict(dict_targets)

        dict_targets = dict(self.target_critic.named_parameters())
        
        for name, weight in self.critic.named_parameters():
            if name in dict_targets:
                dict_targets[name].data.copy_(weight.data*tau + dict_targets[name].data*(1-tau))
        self.target_critic.load_state_dict(dict_targets)

        
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)       
      
        
    def save_models(self):
        print('...saving models...')
        self.actor.save_checkpoint
        self.critic.save_checkpoint
        self.target_actor.save_checkpoint
        self.target_critic.save_checkpoint
        

    def load_models(self):
        print('...loading models...')
        self.actor.load_checkpoint
        self.critic.load_checkpoint
        self.target_actor.load_checkpoint
        self.target_critic.load_checkpoint       
        
        
    def choose_action(self, observation, evaluate=False):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor(state)
        if not evaluate:
            actions += T.empty(self.n_actions).normal_(mean=0.0, 
                             std=self.noise).to(self.actor.device)
            #actions += T.randn(self.n_actions).to(self.actor.device) * self.noise
            
        actions = T.clamp(actions, self.min_action, self.max_action)
        action = np.array([actions.item()])
        
        return action
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)
            
        states = T.tensor(state, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        actions = T.tensor(action, dtype=T.float).to(self.actor.device)            
        rewards = T.tensor(reward, dtype=T.float).to(self.actor.device)
        
        
        target_actions = self.target_actor(states_)
        critic_value_ = T.squeeze(self.target_critic(states_, 
                                                     target_actions), 1)
        
        critic_value = T.squeeze(self.critic(states, actions), 1)
        target = rewards + self.gamma * critic_value_.detach().numpy() * (1 - done)
        loss = nn.MSELoss()
        critic_loss = loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        

        new_policy_actions = self.actor(states)
        actor_loss = -self.critic(states, new_policy_actions)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()          
    
        
if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    agent = Agent(input_dims=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0])
    n_games = 1
    
#    figure_file = 'plots/pendulum.png'
    
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = True
    
    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False
        
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()  
                
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)
#    x = [i+1 for i in range(n_games)]
#    plot_learning_curve(x, score_history, figure_file)
           
                                                
                
                  
    
                
                    
                        
        
        
        
        