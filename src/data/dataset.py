import torch
from torch.utils.data import Dataset

class DrugDataset(Dataset):
    def __init__(self,data):
        self.data=data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        item = self.data[idx]
        return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "target": torch.tensor(item["target"], dtype=torch.float),
                "weight": torch.tensor(item["weight"], dtype=torch.float),
        }
