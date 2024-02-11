# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 07:43:26 2020

@author: andrew.whitworth
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import gym

class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_dims), 
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), 
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_       
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]        

        return states, actions, rewards, states_, terminal
    
    
def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = keras.Sequential([
            keras.layers.Dense(fc1_dims, activation='relu'),
            keras.layers.Dense(fc2_dims, activation='relu'),
            keras.layers.Dense(n_actions, activation=None)])
    
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    
    return model


class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, 
                  input_dims, epsilon_dec=0.9, epsilon_end=0.01, 
                  mem_size=1000000, replace_target=100, fname='dqn_model.h5'):
        
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)
        self.q_targ = build_dqn(lr, n_actions, input_dims, 256, 256)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
        
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
            
        return action
    
    def learn(self):
        if self.memory.mem_cntr > self.batch_size:

            states, actions, rewards, states_, dones = \
                                self.memory.sample_buffer(self.batch_size)
                                           
            #predict Q from evaluation network for current state	
            #    make a copy to later apply update rule
            q_pred = self.q_eval.predict(states)
            q_target = np.copy(q_pred)            
            
            #get next Q from target network for next state
            q_next = self.q_targ.predict(states_)
            
            #get next Q from evaluation network for next state
                #get max action from next Q
            q_eval = self.q_eval.predict(states_)        
            max_actions = np.argmax(q_eval, axis=1).astype(int)

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            
            #establish current Q target value for action taken, using	
                #reward + next Q (target network) from max action (eval network)
            q_target[batch_index, actions] = rewards + \
                self.gamma * q_next[batch_index, max_actions]*dones
            
            #train evaluation network with current state and new target value
            losses = self.q_eval.train_on_batch(states, q_target)
            
            if self.memory.mem_cntr % self.replace_target == 0:                
                self.update_network()
                
            return losses
                

    def update_epsilon(self):
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
            self.epsilon_min else self.epsilon_min 
        
    def update_network(self):  
            self.q_targ.set_weights(self.q_eval.get_weights()) 
        
   
    def save_model(self):
        self.q_eval.save(self.model_file)
            
    def load_model(self):
        self.q_eval = load_model(self.model_file) 
        
        if self.epsilon <= self.epsilon_min:
            self.update_network()
        
        
        
if __name__ == '__main__':
     tf.compat.v1.disable_eager_execution()
     env = gym.make('LunarLander-v2')
     lr = 0.001
     n_games = 500
     agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, 
                   input_dims=env.observation_space.shape, 
                   n_actions=env.action_space.n, mem_size=1000000, 
                   batch_size=64, epsilon_end=0.01)
     losses = []     
     scores = []
     eps_history = []
     for i in range(n_games):
         done = False
         score = 0
         observation = env.reset()
         while not done:
             action = agent.choose_action(observation)
             observation_, reward, done, info = env.step(action)
             score += reward
             agent.store_transition(observation, action, reward, observation_, 
                                    done)
             observation = observation_
             loss = agent.learn()
         
         losses.append(loss)       
         eps_history.append(agent.epsilon)
         scores.append(score)
        
         avg_score = np.mean(scores[-100:])
         avg_loss = np.mean(losses[-100:])
         print('episode ', i, 'score %.1f' % score,
               'avg_score %.1f' % avg_score,
               'avg_loss %.1f' % avg_loss,
               'epsilon %.2f' % agent.epsilon)  
         agent.update_epsilon()  
             
     agent.save_model() 
#         filename = 'lunarlander_tf2.png'
#         x = [i+1 for i in range(n_games)]
#         plot_learning_curve(x, scores, eps_history, filename)
                
        
        
        
        
        
        
        
        
        
        