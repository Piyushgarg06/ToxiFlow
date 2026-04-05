import torch
import torch.nn as nn
from transformers import AutoModel
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    
    def forward(self,input_ids, attention_mask):
        outputs = self.model(input_ids = input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


