# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 08:23:29 2020

@author: andrew.whitworth
"""

import numpy as np
import gym



class CartPoleDelta:
    
    def __init__(self):
        self.CartPole = gym.make('CartPole-v0')      
        self.states = []
        self.poleThetaSpace = np.linspace(-0.209, 0.209, 10)
        self.poleThetaVelSpace = np.linspace(-4, 4, 10)
        self.cartPosSpace = np.linspace(-2.4, 2.4, 10)
        self.cartVelSpace = np.linspace(-4, 4, 10)
        for i in range(len(self.cartPosSpace)+1):
            for j in range(len(self.cartVelSpace)+1):
                for k in range(len(self.poleThetaSpace)+1):
                    for l in range(len(self.poleThetaVelSpace)+1):
                        self.states.append((i,j,k,l))
                        

    def get_state(self, observation):
        cartX, cartXdot, poleTheta, poleThetaDot = observation
        cartX = int(np.digitize(cartX, self.cartPosSpace))
        cartXdot = int(np.digitize(cartXdot, self.cartVelSpace))
        poleTheta = int(np.digitize(poleTheta, self.poleThetaSpace))
        poleThetaDot = int(np.digitize(poleThetaDot, self.poleThetaVelSpace))
        
        return cartX, cartXdot, poleTheta, poleThetaDot 
        

class QAgent:
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.env = CartPoleDelta()
        self.alpha = alpha
        self.gamma  = gamma
        self.epsilon = epsilon
        self.scores = []
        self.Q = np.zeros([11,11,11,11,2])
        

    def choose_action(self, obs, n_actions=2):
        state = self.env.get_state(obs)
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(n_actions)])
        else:
            action_values = [self.Q[state + (a,)] for a in range(n_actions)]
            action = np.argmax(action_values)                
        return action
    
    
    def play(self, n_episodes = 5000):
        for i in range(n_episodes):            
            done = False
            score = 0
            
            obs = self.env.CartPole.reset()
            s = self.env.get_state(obs)
                        
            while not done:
                a = self.choose_action(obs)
                obs, reward, done, info = self.env.CartPole.step(a) 
                score += reward
                
#                if i % 1000 == 0:
#                    self.env.CartPole.render()
                    
                sa = s + (a,)
                s_ = self.env.get_state(obs)
                a_ = np.argmax( [self.Q[s_+(actions,)] for actions in range(2)])
                sa_ = s_ + (a_,)
                
                self.Q[sa] += self.alpha*(reward + self.gamma*self.Q[sa_] - self.Q[sa])
                
                s = s_
                
                
            self.scores.append(score)
            avg_score = np.mean(self.scores[-1000:])
            self.epsilon = self.epsilon - 1 / n_episodes if self.epsilon > 0 else 0
            if i % 1000 == 0:
                print('episode ', i, 'avg_score %.1f' % avg_score, 
                      'epsilon %.2f' % self.epsilon)                    
            if i == n_episodes-1:
                print('episode ', i, 'avg_score %.1f' % avg_score, 
                      'epsilon %.2f' % self.epsilon)        

if __name__ == "__main__":

    alpha = 0.1
    gamma = 0.9    
    epsilon = 1.0

    n_episodes = 5000
    
    Agent = QAgent(alpha, gamma, epsilon)
    
    Agent.play(n_episodes)
    
  #  Agent.play(1)

    
    
    
    
        
        
        
        
    
        
    