from rdkit import Chem
import math

def clean_data(data):
    cleaned=[]
    seen = set()

    for item in data:
        if item["label"] is None or (isinstance(item['label'], float) and math.isnan(item['label'])):
            continue
        if Chem.MolFromSmiles(item["smiles"]) is None:
            continue
        if item["smiles"] in seen:
            continue
        seen.add(item["smiles"])
        cleaned.append(item)
    return cleaned