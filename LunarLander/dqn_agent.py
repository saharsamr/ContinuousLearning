from network import DeepNetwork
from experience_replay import ReplayBuffer

import random
import numpy as np

import torch


class Agent():
    
    def __init__(self, env, actions, epsilon, alpha, discount, buffer_size=10000, batch_size=100, episodes=100):
        
        self.dqn = DeepNetwork(8, 4)
        self.memory = ReplayBuffer(buffer_size)
        self.env = env
        self.epsilon = epsilon
        self.episodes = episodes
        self.alpha = alpha
        self.discount = discount
        self.actions = actions
        self.batch_size = batch_size
        self.optim = torch.optim.Adam(self.dqn.parameters(), lr=self.alpha)
        self.loss_func = torch.nn.SmoothL1Loss()
        
    def select_action(self, state):
        
        if random.random() < self.epsilon:
            return int(np.random.choice(self.actions))
        else:
            return int(torch.argmax(self.dqn(torch.tensor(state))))
    
    def batch_train(self):
        
        batch = self.memory.sample(self.batch_size)
        states = torch.tensor([b['state'] for b in batch])
        actions = torch.tensor([b['action'] for b in batch])
        rewards = torch.tensor([b['reward'] for b in batch])
        next_states = torch.tensor([b['next_state'] for b in batch])
        dones = torch.tensor([1 if b['done'] else 0 for b in batch])
        
        self.optim.zero_grad()
        
        x = self.dqn(states)
        x = torch.tensor([x[a] for x, a in zip(x, actions)])
        y = rewards + torch.mul((self.discount*self.dqn(next_states).max(1).values.unsqueeze(1)), 1 - dones)
        
        loss = self.loss_func(x, y)
        loss.backward()
        self.optim.step()
        
        
    def train(self):
        
        rewards = []
        for episode in range(self.episodes):
            
            state = self.env.reset()
            if episode%100 == 0:
                print(f'episode: {episode}')
            
            episode_rewards = 0
            while True:
                a = self.select_action(state)
                next_state, reward, done, _ = self.env.step(a)
                episode_rewards += reward
                
                self.memory.add_trial(state, a, reward, next_state, done)
                
                if len(self.memory.buffer) > self.batch_size:
                    self.batch_train()
                    
                state = next_state
#                 self.env.render()
                if done:
                    break
                    
            rewards.append(episode_rewards)
            self.epsilon *= 0.99
            
        return rewards