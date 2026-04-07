# 🧪 ToxiFlow

> **⚠️ Work In Progress** — This project is actively under development. Features and APIs may change.

A modular deep learning pipeline for **drug toxicity prediction** using molecular SMILES representations and transformer-based models.

## Overview

ToxiFlow leverages **ChemBERTa** (a chemistry-focused transformer) to predict toxicity labels from the [Tox21 dataset](https://tripod.nih.gov/tox21/). The pipeline handles everything from raw data ingestion to model inference.

## Project Structure

```
drug_pipeline/
├── data/                   # Raw datasets (gitignored)
├── src/
│   ├── data/
│   │   ├── loader.py       # CSV ingestion with column validation
│   │   ├── cleaner.py      # SMILES validation & deduplication
│   │   ├── formatter.py    # Class weight balancing
│   │   ├── dataset.py      # PyTorch Dataset wrapper
│   │   └── dataLoader.py   # PyTorch DataLoader factory
│   ├── models/
│   │   ├── backbone.py     # ChemBERTa encoder
│   │   ├── pooling.py      # Attention-masked mean pooling
│   │   ├── head.py         # Linear prediction head
│   │   └── model.py        # End-to-end DrugModel
│   ├── representations/
│   │   └── tokenizer.py    # SMILES tokenization logic
│   ├── evaluation/
│   │   └── metrics.py      # Evaluation metrics (TBD)
│   ├── trainer/
│   │   ├── loss.py         # Weighted BCE loss implementation
│   │   └── training.py     # Batch training loop with tqdm
│   └── pipeline/
│       └── pipeline.py     # Full pipeline orchestration
├── main.py
├── requirements.txt
└── .gitignore
```

## Tech Stack

- **PyTorch** — Model architecture & training
- **Transformers** (HuggingFace) — ChemBERTa pretrained backbone
- **RDKit** — Molecular SMILES validation
- **Pandas** — Data loading & manipulation

## Getting Started

```bash
# Clone the repo
git clone https://github.com/Piyushgarg06/ToxiFlow.git
cd ToxiFlow

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Roadmap

- [x] Data loading & cleaning pipeline
- [x] Data formatting with class weight balancing
- [x] PyTorch Dataset implementation
- [x] Transformer backbone (ChemBERTa)
- [x] Mean pooling & prediction head
- [x] End-to-end model assembly
- [x] Training loop with weighted loss
- [x] Pipeline orchestration
- [ ] Evaluation metrics (AUC-ROC, Sensitivity, Specificity)

## License

TBD
