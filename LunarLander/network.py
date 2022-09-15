import torch.nn as nn
import torch.nn.functional as F


class DeepNetwork (nn.Module):
    
    def __init__(self, input_d, output_d):
        
        super(DeepNetwork, self).__init__()
        
        self.dropout = nn.Dropout(0.2)
        self.ln1 = nn.Linear(input_d, 15)
        self.ln2 = nn.Linear(15, 10)
        self.output = nn.Linear(10, 4)
        
    def forward(self, state):
        
        x = F.relu(self.ln1(state))
        x = self.dropout(x)
        x = F.relu(self.ln2(x))
        x = self.dropout(x)
        out = self.output(x)
        
        return out
    
    def __save__(self):
        pass
    
    def __load__(self, path):
        pass