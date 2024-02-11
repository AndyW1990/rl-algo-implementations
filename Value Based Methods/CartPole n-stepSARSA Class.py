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
        

class nSarsaAgent:
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, n=10):
        self.env = CartPoleDelta()
        self.alpha = alpha
        self.gamma  = gamma
        self.epsilon = epsilon
        self.n = n
        self.scores = []
        self.Q = {}
        for s in self.env.states:
            for a in range(2):
                self.Q[(s, a)] = 0.0       
           
    
    def choose_action(self, obs, n_actions=2):
        state = self.env.get_state(obs)
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(n_actions)])
        else:
            action_values = [self.Q[(state, a)] for a in range(n_actions)]
            action = np.argmax(action_values)                
        return action
    
    
    def update_q(self, T, tau, obs, state_memory, action_memory, reward_memory):
        G = [self.gamma**(j-tau-1)*reward_memory[j%self.n] \
             for j in range(tau+1, min(tau+self.n, T)+1)] 
        G = np.sum(G)
        if tau + self.n < T:
            s = self.env.get_state(state_memory[(tau+self.n)%self.n])
            a = int(action_memory[(tau+self.n)%self.n])
            G += self.gamma**self.n * self.Q[(s,a)]
        s = self.env.get_state(state_memory[tau%self.n])
        a = action_memory[tau%self.n]
        self.Q[(s,a)] += self.alpha*(G - self.Q[(s,a)]) 
        
    
    def play(self, n_episodes = 5000):
        for i in range(n_episodes):            
            done = False
            score = 0
            t = 0 
            T = np.inf
            state_memory = np.zeros([self.n, 4])
            action_memory = np.zeros([self.n])
            reward_memory = np.zeros([self.n])                 
            obs = self.env.CartPole.reset()
            action = self.choose_action(obs)
            action_memory[t%self.n] = action
            state_memory[t%self.n] = obs     
            while not done:
                obs, reward, done, info = self.env.CartPole.step(action) 
                score += reward
                state_memory[(t+1)%self.n] = obs
                reward_memory[(t+1)%self.n] = reward
                
                if done:
                    T = t + 1
                action = self.choose_action(obs)
                action_memory[(t+1)%self.n] = action
                tau = t - self.n + 1
                
                if tau >= 0:
                    self.update_q(T, tau, obs, state_memory, action_memory, reward_memory)
                
                t+=1
                
            for tau in range(t-self.n+1, T):
                self.update_q(T, tau, obs, state_memory, action_memory, reward_memory)
                
                
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

    n = 10
    n_episodes = 5000
    
    Agent = nSarsaAgent(alpha, gamma, epsilon, n)
    
    Agent.play(n_episodes)
    
    states = Agent.env.states
    
    
    

                
        
        
        
        
        
    
        
    