import torch
import torch.nn as nn

loss_fn = nn.BCEWithLogitsLoss(reduction='none')

def compute_loss(logits, target, weight):
    target = target.unsqueeze(-1)
    weight = weight.unsqueeze(-1)
    loss = loss_fn(logits, target)
    loss = loss * weight
    loss = loss.mean()
    return loss

