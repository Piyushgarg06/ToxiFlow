from .backbone import Backbone
from .head import Head
from .pooling import MeanPooling
import torch
import torch.nn as nn

class DrugModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.pooling = MeanPooling()
        self.head = Head()
    def forward(self,input_ids,attention_mask):
        hidden_states = self.backbone(input_ids, attention_mask)
        pooled = self.pooling(hidden_states, attention_mask)
        output = self.head(pooled)
        return output 

