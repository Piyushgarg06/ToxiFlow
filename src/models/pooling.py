import torch
import torch.nn as nn 

class MeanPooling(nn.Module):
    def forward(self, hidden_states, attention_mask):
        expanded_mask = attention_mask.unsqueeze(-1).float()
        masked_hidden = hidden_states*expanded_mask
        
        mask_sum = expanded_mask.sum(dim=1)
        sum_hidden = masked_hidden.sum(dim=1)

        mean = sum_hidden/mask_sum
        return mean
