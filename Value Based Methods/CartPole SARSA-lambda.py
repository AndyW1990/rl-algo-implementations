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
        

class lSarsaAgent:
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, lambd=0.9, trace='accumulating', states_to_cache=100):
        self.env = CartPoleDelta()
        self.alpha = alpha
        self.gamma  = gamma
        self.epsilon = epsilon
        self.lambd = lambd
        self.trace = trace
        self.scores = []
        self.Q = np.zeros([11,11,11,11,2])
        self.states_to_cache = states_to_cache

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
            t = 0
            
            E = np.zeros_like(self.Q)
            
            
            self.state_memory = np.zeros([self.states_to_cache,4])
            self.action_memory = np.zeros([self.states_to_cache])
            self.reward_memory = np.zeros([self.states_to_cache])               

            obs = self.env.CartPole.reset()
            s = self.env.get_state(obs)
            a = self.choose_action(obs)

            self.state_memory[t%self.states_to_cache,:] = s           
            self.action_memory[t%self.states_to_cache] = a
            self.reward_memory[t%self.states_to_cache] = 0
            
            # Render every x episodes
            # if i % 1000 == 0:
            #     self.env.CartPole.render()
                
            while not done:
                t += 1
                obs, reward, done, info = self.env.CartPole.step(a) 
                score += reward
                

                
                s_ = self.env.get_state(obs)
                a_ = self.choose_action(obs)
 
                self.state_memory[(t+1)%self.states_to_cache,:] = s_           
                self.action_memory[(t+1)%self.states_to_cache] = a_
                self.reward_memory[(t+1)%self.states_to_cache] = reward
                
                sa = s + (a,)
                sa_ = s_ + (a_,)
                temp_diff = reward + self.gamma*self.Q[sa_] - self.Q[sa]
                
                if self.trace == 'accumulating':
                    E[sa] += 1
                elif self.trace == 'dutch':
                    E[sa] = (1-self.alpha)*E[sa] + 1 
                else:
                    E[sa] = 1
                
                self.Q += self.alpha*temp_diff*E
                E *= self.alpha*self.lambd

                s = s_
                a = a_
                
                
            self.scores.append(score)
            avg_score = np.mean(self.scores[-100:])
            self.epsilon = self.epsilon - 1 / n_episodes if self.epsilon > 0 else 0
            if i % 100 == 0:
                print('episode ', i, 'avg_score %.1f' % avg_score, 
                      'epsilon %.2f' % self.epsilon)                    
            if i == n_episodes-1:
                print('episode ', i, 'avg_score %.1f' % avg_score, 
                      'epsilon %.2f' % self.epsilon)        

if __name__ == "__main__":

    alpha = 0.1
    gamma = 0.9    
    epsilon = 1.0

    lambd = 0.9
    trace = 'accumulating'
    n_episodes = 5000
    
    Agent = lSarsaAgent(alpha, gamma, epsilon, lambd, trace)
    
    Agent.play(n_episodes)
    
    
    
    
    
        
        
        
        
    
        
    