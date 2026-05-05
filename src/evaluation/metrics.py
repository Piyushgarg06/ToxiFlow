import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix


def evaluate(model, dataloader):
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            target = batch['target']

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_targets.extend(target.numpy().flatten())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    preds = (all_probs >= 0.5).astype(int)

    auroc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.0
    accuracy = accuracy_score(all_targets, preds)
    sensitivity = recall_score(all_targets, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_targets, preds, labels=[0,1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"AUC-ROC: {auroc:.4f} | Accuracy: {accuracy:.4f} | Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f}")

    return {"auroc": auroc, "accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity}
