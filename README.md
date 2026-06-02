# Citation Intent Classifier with SciBERT

This project implements a modular citation intent classifier using **SciBERT** (`allenai/scibert_scivocab_uncased`) as the base model, with three distinct pipelines for training and evaluation on:
1. **SciCite** (3 classes)
2. **ACL-ARC** (6 classes)
3. **SOFT** framework dataset (customizable classes)

The training scripts support options optimized for running on CPU laptop environments by allowing you to freeze SciBERT's weights and only train the classification head.

---

## Project Structure

```
citcl/
│
├── pyproject.toml              # Project dependencies & configuration
├── requirements.txt            # Package list for easy installation
├── README.md                   # This file
│
├── src/                        # Core codebase
│   ├── config.py               # Shared training configurations (dataclasses/argparse)
│   ├── data.py                 # Data loaders and preprocess pipeline for SciCite, ACL-ARC, and SOFT
│   ├── model.py                # Classifier module wrapping SciBERT and classification head
│   └── trainer.py              # PyTorch training, evaluation, checkpointing, and metric logging
│
└── pipelines/                  # Pipeline execution scripts
    ├── train_scicite.py        # SciCite pipeline execution
    ├── train_acl_arc.py        # ACL-ARC pipeline execution
    └── train_soft.py           # SOFT framework pipeline execution
```

---

## Installation

Ensure you have Python 3.8+ installed. You can install all requirements with:

```bash
pip install -r requirements.txt
```

---

## Running the Pipelines

### 1. SciCite Pipeline
*   **Quick Test (Frozen SciBERT)**:
    ```bash
    python pipelines/train_scicite.py --quick_test --freeze_bert
    ```
*   **Full Fine-tuning**:
    ```bash
    python pipelines/train_scicite.py
    ```

### 2. ACL-ARC Pipeline
*   **Quick Test (Frozen SciBERT)**:
    ```bash
    python pipelines/train_acl_arc.py --quick_test --freeze_bert
    ```
*   **Full Fine-tuning**:
    ```bash
    python pipelines/train_acl_arc.py
    ```

### 3. SOFT Pipeline
*   **Quick Test (Frozen SciBERT)**:
    ```bash
    python pipelines/train_soft.py --quick_test --freeze_bert
    ```
*   **Full Fine-tuning (with your own local data file)**:
    ```bash
    python pipelines/train_soft.py --data_path path/to/SOFT_dataset.json
    ```

---

## Custom Settings and Hyperparameters

You can pass command-line arguments to customize your run:

*   `--freeze_bert`: Freeze SciBERT base weights (highly recommended to run quickly in seconds/minutes on CPU).
*   `--epochs <num>`: Set the number of epochs (default is `3` for fine-tuning, `10` for frozen runs).
*   `--lr <rate>`: Learning rate (default `2e-5` for full tuning, `1e-3` for frozen runs).
*   `--batch_size <size>`: Batch size (default `16`).
*   `--max_length <length>`: Maximum sequence token length (default `128`).
*   `--quick_test`: Load a small subset of the data (50 samples) and train for 1 epoch.
*   `--seed <seed>`: Random seed for reproducibility.

All logs, checkpoints, classification reports, and evaluation metrics are saved to the `outputs/` directory.
