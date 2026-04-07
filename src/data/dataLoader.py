from torch.utils.data import DataLoader
try:
    from .dataset import DrugDataset
except ImportError:
    from dataset import DrugDataset

def get_loader(data,batch_size=32, shuffle=True):
    dataset = DrugDataset(data)
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataLoader