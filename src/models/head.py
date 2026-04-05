import torch 
import torch.nn as nn 
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, 1)
    def forward(self,x):
        return self.l1(x)


