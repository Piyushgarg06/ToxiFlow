from loader import load_data
from cleaner import clean_data

def format_data(data):
    formatted=[]
    total = len(data)
    count_0 = sum(1 for d in data if d["label"]==0)
    count_1 = sum(1 for d in data if d["label"]==1)

    weight_0 = total/count_0
    weight_1 = total/count_1

    for item in data:
        if item['label']==0:
            new_item = {
                "input": item['smiles'],
                "target": item['label'],
                "weight": weight_0
            }
        else:
            new_item={
                "input": item['smiles'],
                "target": item['label'],
                "weight": weight_1
        }
        formatted.append(new_item)
    return formatted
    