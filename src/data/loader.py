import pandas as pd 
from pathlib import Path

df = pd.read_csv("data/tox21.csv")
path = "data/tox21.csv"
def load_data(path,smiles_cols,target_cols):
    path = Path(path)
    
    if path.exists():
        df = pd.read_csv(path)
    else:
        raise FileNotFoundError("path is either broken or DNE")
    required_cols = {smiles_cols, target_cols}
    missing = [cols for cols in required_cols if cols not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[[smiles_cols, target_cols]]
    df = df.rename(columns={
        smiles_cols:"smiles",
        target_cols:"label"
    })
    data = df.to_dict(orient="records")
    return data


