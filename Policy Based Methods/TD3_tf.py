# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:05:38 2021

@author: andrew.whitworth
"""

import gym
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


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
    
    
    
class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=2, name='actor', 
                 chkpt_dir='TD3_files/'):
        super(ActorNetwork, self).__init__()
        
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                                            self.model_name + '_td3.h5')
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')
        
    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        
        # if action bounds not +/- 1, can multiply here
        mu = self.mu(prob)
        
        return mu
     
        
class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, name='critic', 
                 chkpt_dir='TD3_files/'):
        super(CriticNetwork, self).__init__()
        
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                                            self.model_name + '_td3.h5')
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)
        
    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        
        q = self.q(action_value)
        
        return q


    
class Agent():
    def __init__(self, input_dims, alpha=0.001, beta=0.001, env=None,
                 gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 fc1=512, fc2=512, batch_size=100, start_steps=10000, 
                 update_every=1, noise=0.2, noise_clip=0.5,
                 policy_delay=2):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.start_steps = start_steps
        self.update_every = update_every
        
        
        self.noise = noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.env = env
        
        self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.critic_1 = CriticNetwork(name='critic_1')            
        self.critic_2 = CriticNetwork(name='critic_2')      
        
        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor')
        self.target_critic_1 = CriticNetwork(name='target_critic_1')  
        self.target_critic_2 = CriticNetwork(name='target_critic_2')  
        
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))        
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta))
        
        self.update_network_parameters(tau=1)        

        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
            
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)
        
        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_critic_1.set_weights(weights)    
 
        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_critic_2.set_weights(weights)    
        
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def save_models(self):
        print('...saving models...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)    
        self.critic_2.save_weights(self.critic_2.checkpoint_file)           
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoint_file) 
        self.target_critic_2.save_weights(self.target_critic_2.checkpoint_file)         

    def load_models(self):
        print('...loading models...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)   
        self.critic_2.load_weights(self.critic_2.checkpoint_file)           
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.load_weights(self.target_critic_1.checkpoint_file) 
        self.target_critic_2.load_weights(self.target_critic_2.checkpoint_file) 
        
    def choose_action(self, observation, evaluate=False, j=np.inf):
     
        if j > self.start_steps:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            actions = self.actor(state)
            if not evaluate:
                actions += tf.random.normal(shape=[self.n_actions], mean=0.0,
                                            stddev=self.noise)
                
            actions = tf.clip_by_value(actions, self.min_action, self.max_action)[0]
        
        else:
            actions = self.env.action_space.sample()
        

        return actions
            
        
    def learn(self, j=False):
        
        if not j: 
            j = self.start_steps
            self.memory.mem_cntr = self.start_steps
            
        if self.memory.mem_cntr < self.start_steps or j % self.update_every != 0:
            return
        
        for i in range(self.update_every):
            state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)
                
            states = tf.convert_to_tensor(state, dtype=tf.float32)
            states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
            actions = tf.convert_to_tensor(action, dtype=tf.float32)
            reward = tf.convert_to_tensor(reward, dtype=tf.float32)
            
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                clipped_noise = tf.clip_by_value(tf.random.normal(
                                    shape=[self.n_actions], mean=0.0,
                                    stddev=self.noise),-self.noise_clip, 
                                    self.noise_clip)
                target_actions = tf.clip_by_value(self.target_actor(states_) + \
                                   clipped_noise, self.min_action, self.max_action)
                                
                critic_value_1_ = tf.squeeze(self.target_critic_1(states_, 
                                                             target_actions), 1)
                critic_value_2_ = tf.squeeze(self.target_critic_2(states_, 
                                                             target_actions), 1)
                
                critic_value_ = tf.minimum(critic_value_1_,critic_value_2_)
                
                critic_value_1 = tf.squeeze(self.critic_1(states, actions), 1)
                critic_value_2 = tf.squeeze(self.critic_2(states, actions), 1)
                
                target = reward + self.gamma * critic_value_ * (1 - done)
                
                critic_1_loss = keras.losses.MSE(target, critic_value_1)
                critic_2_loss = keras.losses.MSE(target, critic_value_2)
                
                critic_1_network_gradient = tape1.gradient(critic_1_loss, 
                                                    self.critic_1.trainable_variables)
                critic_2_network_gradient = tape2.gradient(critic_2_loss, 
                                                    self.critic_2.trainable_variables)
                
            self.critic_1.optimizer.apply_gradients(zip(critic_1_network_gradient, 
                                                self.critic_1.trainable_variables))
            
            self.critic_2.optimizer.apply_gradients(zip(critic_2_network_gradient, 
                                                self.critic_2.trainable_variables))        

            if j % self.policy_delay == 0:
                
                with tf.GradientTape() as tape:
                    new_policy_actions = self.actor(states)
                    actor_loss = -self.critic_1(states, new_policy_actions)
                    actor_loss = tf.math.reduce_mean(actor_loss)
                    
                actor_network_gradient = tape.gradient(actor_loss, 
                                                       self.actor.trainable_variables)
                
                
                self.actor.optimizer.apply_gradients(zip(actor_network_gradient, 
                                                    self.actor.trainable_variables))
                
                self.update_network_parameters()
            
            
if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(input_dims=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0])
    n_games = 300
    
#    figure_file = 'plots/pendulum.png'
    
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    evaluate = False
    
    
    j = 0
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
        j = agent.start_steps
    
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn(j)
            observation = observation_
            j += 1
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            if not evaluate:
                agent.save_models()  
                
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)
    env.close()
#    x = [i+1 for i in range(n_games)]
#    plot_learning_curve(x, score_history, figure_file)            
                