# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 08:23:46 2020

@author: andrew.whitworth
"""

from DQN_np import Agent
import numpy as np
import gym



train = True

env = gym.make('LunarLander-v2')
lr = 0.001
n_games = 500 if train else 5
epsilon = 1.0 if train else 0.0

agent = Agent(lr=lr, gamma=0.99, n_actions=env.action_space.n, 
              epsilon=epsilon, batch_size=64, input_dims=env.observation_space.shape,
              epsilon_dec=0.9, epsilon_end=0.01,  mem_size=1000000, 
              replace_target=100, fname='weights_2')
scores = []
eps_history = []

if not train:
    agent.load_weights() 

for i in range(n_games):
    done = False
    score = 0
    observation = env.reset()
    while not done:
        #if not train:             
        #    env.render()
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, 
                               done)
        observation = observation_
        loss = agent.learn()
        
    eps_history.append(agent.epsilon)
    scores.append(score)
   
    avg_score = np.mean(scores[-100:])
    avg_loss = np.mean(loss)
    print('episode ', i, 'score %.1f' % score,
          'avg_score %.1f' % avg_score,
          'mean_loss %.1f' % avg_loss,
          'epsilon %.2f' % agent.epsilon)  
    agent.update_epsilon()  
    
if train: 
    agent.save_weights()
