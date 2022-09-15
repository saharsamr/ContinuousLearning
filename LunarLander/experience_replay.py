import numpy as np


class ReplayBuffer():
    # TODO: Define a class for storing states, actions, and rewards.
    # It is suggested to label terminal states.
    #
    # Hints: 1- This class should contain 2 methods. One for storing
    #           transtions, and the other for sampling a batch of 
    #           experiences.
    
    def __init__(self, buffersize=10000):
        
        self.buffer_size = buffersize
        self.buffer = []
        
    def add_trial(self, state, action, reward, next_state, done):
        
        if (self.buffer_size and len(self.buffer) < self.buffer_size) or not self.buffer_size:
            self.buffer.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
        else:
            self.buffer
            
    def sample(self, batch_size):
        
        return np.random.choice(self.buffer, size=batch_size)