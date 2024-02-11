# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:02:57 2020

@author: andrew.whitworth
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import numpy as np
import gym


class PolicyGradientNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(PolicyGradientNetwork, self).__init__()
        
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.pi = Dense(self.n_actions, activation='softmax')
        
        
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        
        pi = self.pi(x)
        
        return pi
        

class Agent:
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=4, fc1_dims=256, fc2_dims=256):
        self.gamma = gamma
        self.alpha = alpha
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy = PolicyGradientNetwork(n_actions=n_actions)
        self.policy.compile(optimizer=Adam(learning_rate=self.alpha))        
        
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        probs = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()
        return action.numpy()[0]
    
    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        
    def learn(self):
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rewards = np.array(self.reward_memory)
        
        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k]*discount
                discount *= self.gamma
            G[t] = G_sum
            
            
        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss -= g * tf.squeeze(log_prob)
                
        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))
        
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
                        
                    
                    
if __name__ == '__main__':
    
    env = gym.make('LunarLander-v2')
    agent = Agent(alpha=0.0005, gamma=0.99, n_actions=env.action_space.n)

    n_episodes = 1
    
    score_history = []
    
    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward
        score_history.append(score)
        
        agent.learn()
        
        avg_score = np.mean(score_history[-100:])
        print('episode ', i, 
              'score %.1f' % score,
              'avg_score %.1f' % avg_score)  
    
    
    
    
    
    
    
    
                     
            
            
                    
         