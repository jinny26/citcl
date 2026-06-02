# Implementation Plan - Citation Intent Classifier with SciBERT

This document details the plan to build a citation intent classification system using **SciBERT** (`allenai/scibert_scivocab_uncased`) as the base model, with separate training and evaluation pipelines for multiple datasets: **SciCite**, **ACL-ARC**, and **SOFT**.

---

## CPU Performance & Optimization

Since CUDA is not available on this system, we will optimize for CPU training:

1. **Option A: Full Fine-tuning** (standard)
   - **Time Estimate**: ~1.5 hours total for all three datasets on CPU.
     - SciCite (~11k samples): ~10–12 min/epoch.
     - ACL-ARC (~2k samples): ~2 min/epoch.
     - SOFT (~2k samples): ~2 min/epoch.
   - **Ideal Hyperparameters**:
     - Learning Rate: `2e-5` (using AdamW with linear warmup)
     - Batch Size: `16` (keeps memory footprint low)
     - Epochs: `3`
     - Max Sequence Length: `128` (highly recommended for CPU speed)

2. **Option B: Feature Extraction / Frozen SciBERT** (extremely fast)
   - **Time Estimate**: < 5 minutes total. We freeze SciBERT's weights, pre-compute the embeddings (using the `CLS` token representation), and train a PyTorch classification head.
   - **Ideal Hyperparameters**:
     - Learning Rate: `1e-3`
     - Batch Size: `32`
     - Epochs: `10`
     - Freeze base: `--freeze_bert` flag

We will implement support for both fine-tuning options via a command-line flag (`--freeze_bert`), allowing you to choose between full fine-tuning and fast classification-head training.

---

## Dataset Loading Strategy

1. **SciCite**: Loaded directly from Hugging Face: `allenai/scicite`.
2. **ACL-ARC**: Loaded directly from Hugging Face: `kejian/ACL-ARC`.
3. **SOFT**: The user will provide the dataset or we will read it from local files (e.g. `--data_path path/to/dataset.json`), with fallback options to parse it if downloaded.

---

## Proposed Changes

We will create a clean, modular Python codebase under the `citcl` project workspace.

### Directory Structure
```
citcl/
│
├── pyproject.toml              # Project dependencies & packaging configuration
├── README.md                   # Updated instructions on running pipelines
├── requirements.txt            # Package list for easy installation
│
├── src/                        # Main source package
│   ├── __init__.py
│   ├── config.py               # Shared training configurations (dataclasses/argparse)
│   ├── data.py                 # Data loading, tokenization, and pipeline-specific preprocessing
│   ├── model.py                # Classifier module using SciBERT and classification head
│   └── trainer.py              # Common training, evaluation, and logging loop
│
└── pipelines/                  # Pipeline execution scripts
    ├── train_scicite.py        # Pipeline for SciCite dataset
    ├── train_acl_arc.py        # Pipeline for ACL-ARC dataset
    └── train_soft.py           # Pipeline for SOFT dataset
```

---

### Project Configuration & Setup

#### [MODIFY] [pyproject.toml](file:///c:/Users/Jini%20John/projects/citcl/pyproject.toml)
Update the dependencies list to include:
- `transformers`
- `datasets`
- `torch`
- `scikit-learn`
- `pandas`
- `tqdm`

#### [NEW] [requirements.txt](file:///c:/Users/Jini%20John/projects/citcl/requirements.txt)
Add explicit versions for standard python library installations.

---

### Core Source Files

#### [NEW] [config.py](file:///c:/Users/Jini%20John/projects/citcl/src/config.py)
Defines shared settings like model names (`allenai/scibert_scivocab_uncased`), training hyperparameters, output directories, batch sizes, learning rates, and execution settings (e.g., CPU/GPU detection, subset size for quick validation, freezing BERT).

#### [NEW] [data.py](file:///c:/Users/Jini%20John/projects/citcl/src/data.py)
Implements:
- Generic Hugging Face Dataset loaders.
- Specialize loaders for:
  - **SciCite**: Loaded via Hugging Face `datasets.load_dataset("allenai/scicite")`.
  - **ACL-ARC**: Loaded via Hugging Face `datasets.load_dataset("kejian/ACL-ARC")`.
  - **SOFT**: Reads local JSON/JSONL datasets with proper alignment of context and label fields.
- Dataset-specific tokenization and alignment of text (citation context) and labels.

#### [NEW] [model.py](file:///c:/Users/Jini%20John/projects/citcl/src/model.py)
Implements:
- `SciBERTClassifier`: A PyTorch module that wraps SciBERT (`AutoModel.from_pretrained`) and attaches a classification head (Linear/MLP layers) to classify citation intent.
- Option to freeze/unfreeze SciBERT base weights based on `--freeze_bert`.

#### [NEW] [trainer.py](file:///c:/Users/Jini%20John/projects/citcl/src/trainer.py)
Implements:
- Training loops using Hugging Face's `Trainer` API or native PyTorch loop.
- Compute metrics function (Accuracy, Precision, Recall, Macro F1, Micro F1, Confusion Matrix).
- Save/load model checkpoints.

---

## Dataset-Specific Pipelines

### [NEW] [train_scicite.py](file:///c:/Users/Jini%20John/projects/citcl/pipelines/train_scicite.py)
Script to run the SciCite training and evaluation pipeline.

### [NEW] [train_acl_arc.py](file:///c:/Users/Jini%20John/projects/citcl/pipelines/train_acl_arc.py)
Script to run the ACL-ARC training and evaluation pipeline.

### [NEW] [train_soft.py](file:///c:/Users/Jini%20John/projects/citcl/pipelines/train_soft.py)
Script to run the SOFT training and evaluation pipeline.

---

## Verification Plan

### Automated Tests
1. **Sanity Check Execution**:
   Run each pipeline script with a small subset of the data (e.g., `--quick_test` or `--max_samples 50`) and 1 epoch to ensure data download, tokenization, model loading, forward/backward pass, and validation work end-to-end on CPU without errors:
   - `python pipelines/train_scicite.py --quick_test`
   - `python pipelines/train_acl_arc.py --quick_test`
   - `python pipelines/train_soft.py --quick_test`

2. **Metrics Verification**:
   Verify that performance reports (classification report, confusion matrix) are generated and saved correctly in the output directories.
