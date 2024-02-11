# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:29:27 2020

@author: andrew.whitworth
"""

import numpy as np
import pickle

def relu(z):
    return np.maximum(0,z)
 
def relu_grad(dA, z):
    mask = (dA <= 0)
    dA[mask] = 0
    dx = dA
    return dx
      
def l_relu(z):
    return np.maximum(0.01*z,z)
 
def l_relu_grad(dA, z):
    mask = (dA <= 0)
    dA[mask] = 0.01
    dx = dA
    return dx

def tanh(z):
    return np.tanh(z)

def tanh_grad(dA, z):
    A = tanh(z)
    dz = dA * (1 - np.square(A))
    return dz

    
class Regression_Network():
    def __init__(self, in_dims, out_dims, hidden_dims, lr):
        
        #fix this to take tuple or anythting else
        self.layer_dims = in_dims + hidden_dims + (out_dims,)
        self.lr = lr

        self.init_params()
        
        
    def init_params(self):        
        self.parameters = {}
        L = len(self.layer_dims) 
        
        for l in range(1, L):
            
            self.parameters['W' + str(l)] = np.random.rand(self.layer_dims[l],
                            self.layer_dims[l-1])
            self.parameters['b' + str(l)] = np.zeros([self.layer_dims[l],1]) 
    

    def lin_activation_fwd(self, A_prev, W, b, activation='linear'):
        z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b)
        
        if activation == 'relu':
            A = relu(z)  
        elif activation == 'l_relu':
            A = l_relu(z)  
        elif activation == 'tanh':
            A = tanh(z)       
        else:
            A = z
        
        self.caches.append((cache, z))
        return A    
    
    def model_fwd(self, x):
        self.caches = []
        A = x
        L = len(self.parameters) // 2
        
        for l in range(1, L):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            A = self.lin_activation_fwd(A_prev, W, b, activation='tanh')
            
        W = self.parameters['W' + str(L)]
        b = self.parameters['b' + str(L)]
        y_ = self.lin_activation_fwd(A, W, b, activation='linear')
            
        return y_
    
    def loss(self, y_, y):
        m = y.shape[1]
        mse = 1/2/m * np.sum((y_ - y)**2)
        return mse
    
    
    def lin_back(self, dz, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1/m) * np.dot(dz, A_prev.T)
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dz)
        
        return dA_prev, dW, db
    
    def lin_activation_back(self, dA, cache, activation='linear'):
        lin_cache, z = cache
        if activation == 'relu':        
            dz = relu_grad(dA, z)
        elif activation == 'l_relu':        
            dz = l_relu_grad(dA, z)         
        elif activation == 'tanh':
            dz = tanh_grad(dA, z)  
        else:
            dz = dA
        
        dA_prev, dW, db = self.lin_back(dz, lin_cache)            
            
        return dA_prev, dW, db
    
    def model_back(self, y_, y):
        self.grads = {}
        L = len(self.caches)

        dA = -(y - y_)
        cache = self.caches[L-1]
        dA, self.grads['dW' + str(L)], self.grads[
                'db' + str(L)] = self.lin_activation_back(dA, cache, 
                                          activation='linear')

        for l in range(L-1, 0, -1):
            cache = self.caches[l-1]
            dA, self.grads['dW' + str(l)], self.grads[
                    'db' + str(l)] = self.lin_activation_back(dA, 
                                     cache, activation='tanh')
                
           
    def clip_grads(self):
        L = len(self.parameters) // 2  
        for l in range(1, L):
            self.grads['dW' + str(l)][np.where(self.grads['dW' + str(l)] > 1)] = 1
            self.grads['dW' + str(l)][np.where(self.grads['dW' + str(l)] < -1)] = -1 
            self.grads['db' + str(l)][np.where(self.grads['db' + str(l)] > 1)] = 1
            self.grads['db' + str(l)][np.where(self.grads['db' + str(l)] < -1)] = -1                
    
    def update_params(self):
        self.clip_grads()
        L = len(self.parameters) // 2

        for l in range(1, L):
            self.parameters['W' + str(l)] -= self.lr*self.grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= self.lr*self.grads['db' + str(l)]

            
    def train(self, x, y, iterations=1, print_loss=False):
        losses = []
        for i in range(iterations):
                y_ = self.model_fwd(x)    
                
                loss = self.loss(y_, y)
                losses.append(loss)
                
                self.model_back(y_, y)
                self.update_params()
             
                if print_loss:
                    if i % 100 == 0:
                        print(f"The loss after iteration {i} is: {loss:.4f}")

        return losses

class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        
        #need to fix dims, probs reverse the NN and put these back to RL standards
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((*input_dims, self.mem_size), 
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((*input_dims, self.mem_size), 
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[:,index] = state
        self.new_state_memory[:,index] = state_       
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[:,batch]
        states_ = self.new_state_memory[:,batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]        

        return states, actions, rewards, states_, terminal


class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, 
                  input_dims, epsilon_dec=0.9, epsilon_end=0.01, 
                  mem_size=1000000, fname='weights'):
        
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.fname = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = Regression_Network(input_dims, n_actions, (256,256), lr)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
        
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation]).T       
            actions = self.q_eval.model_fwd(state)
            action = np.argmax(actions)
            
        return action
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = \
        self.memory.sample_buffer(self.batch_size)
        
        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)
        
        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        q_target[batch_index, actions] = rewards + \
            self.gamma * np.max(q_next, axis=1)*dones
        
        self.q_eval.train_on_batch(states, q_target)
            
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > \
        self.epsilon_min else self.epsilon_min
                
            
    def load_weights(self):
        with open(self.fname + '.pickle', 'rb') as handle:
            self.q_eval.parameters = pickle.load(handle)

      
        
