from src.data.cleaner import clean_data
from src.data.formatter import format_data
from src.data.loader import load_data
from src.data.dataset import DrugDataset
from src.data.dataLoader import get_loader
from src.models.model import DrugModel
from src.representations.tokenizer import tokenize_data
from src.trainer.training import training
from src.trainer.loss import compute_loss
import torch
import torch.nn as nn
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

path = 'data/tox21.csv'
data = load_data(path,smiles_cols="smiles", target_cols="SR-p53")
data = clean_data(data)
data = format_data(data)
data = tokenize_data(data,tokenizer, max_length=40)
data = DrugDataset(data)
dataLoader = get_loader(data,batch_size=32, shuffle=True)

epochs = 5
learning_rate = 3e-5
model = DrugModel()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(epochs):
    total_loss = training(model, dataLoader, compute_loss, optimizer)
    print(f"epoch : {epoch+1}, loss : {total_loss}")

results = evaluate(model, dataLoader)

