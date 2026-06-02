# Walkthrough - SciBERT Citation Intent Classification Pipelines

We have successfully built the citation intent classification pipelines using SciBERT as the base model. The code is modular, well-structured, and optimized to run on CPU environments by supporting a frozen base model flag (`--freeze_bert`).

---

## Code Base Structure

The codebase is organized as follows:
- **[config.py](file:///c:/Users/Jini%20John/projects/citcl/src/config.py)**: Centralizes all training arguments, hyperparameters, random seeds, and quick check flags.
- **[data.py](file:///c:/Users/Jini%20John/projects/citcl/src/data.py)**: Implements specialized loaders and preprocessing pipelines for SciCite, ACL-ARC, and SOFT datasets. It handles tokenization, label mappings, and dataset splits.
- **[model.py](file:///c:/Users/Jini%20John/projects/citcl/src/model.py)**: Implements `SciBERTClassifier`, wrapping the transformer base and attaching a dropout + linear layer classification head. Supports freezing base weights.
- **[trainer.py](file:///c:/Users/Jini%20John/projects/citcl/src/trainer.py)**: A modular PyTorch training loop that tracks loss, saves the best checkpoint based on validation Macro F1, computes standard classification metrics (Accuracy, Precision, Recall, Macro/Micro F1), and generates reports.
- **[train_scicite.py](file:///c:/Users/Jini%20John/projects/citcl/pipelines/train_scicite.py)**: Executable training script for the SciCite pipeline.
- **[train_acl_arc.py](file:///c:/Users/Jini%20John/projects/citcl/pipelines/train_acl_arc.py)**: Executable training script for the ACL-ARC pipeline.
- **[train_soft.py](file:///c:/Users/Jini%20John/projects/citcl/pipelines/train_soft.py)**: Executable training script for the SOFT framework pipeline.

---

## How to Run the Pipelines

To run the pipelines, open your terminal in the `citcl` directory:

### 1. SciCite Pipeline
*   **Quick Test (Frozen SciBERT)**:
    ```powershell
    python pipelines/train_scicite.py --quick_test --freeze_bert
    ```
*   **Full Fine-tuning (on all data)**:
    ```powershell
    python pipelines/train_scicite.py
    ```

### 2. ACL-ARC Pipeline
*   **Quick Test (Frozen SciBERT)**:
    ```powershell
    python pipelines/train_acl_arc.py --quick_test --freeze_bert
    ```
*   **Full Fine-tuning (on all data)**:
    ```powershell
    python pipelines/train_acl_arc.py
    ```

### 3. SOFT Pipeline
*   **Quick Test (Frozen SciBERT)**:
    ```powershell
    python pipelines/train_soft.py --quick_test --freeze_bert
    ```
*   **Full Fine-tuning (with your own local data file)**:
    ```powershell
    python pipelines/train_soft.py --data_path path/to/SOFT_dataset.json
    ```

---

## Verification Results

All three pipelines were successfully validated using the `--quick_test --freeze_bert` flags. The verification outputs have been saved in the `outputs/` folder.

### 1. SciCite Validation
- **Train Loss**: `0.8509`
- **Val Loss**: `0.8455`
- **Val Macro F1**: `0.2447`
- **Test Accuracy**: `52.0%`
- **Saved Reports**:
  - Results JSON: [scicite_results.json](file:///c:/Users/Jini%20John/projects/citcl/outputs/scicite_results.json)
  - Text Report: [scicite_classification_report.txt](file:///c:/Users/Jini%20John/projects/citcl/outputs/scicite_classification_report.txt)

### 2. ACL-ARC Validation
- **Train Loss**: `1.6756`
- **Val Loss**: `1.2190`
- **Val Macro F1**: `0.1795`
- **Test Accuracy**: `50.0%`
- **Saved Reports**:
  - Results JSON: [acl_arc_results.json](file:///c:/Users/Jini%20John/projects/citcl/outputs/acl_arc_results.json)
  - Text Report: [acl_arc_classification_report.txt](file:///c:/Users/Jini%20John/projects/citcl/outputs/acl_arc_classification_report.txt)

### 3. SOFT Validation
- **Train Loss**: `1.1902`
- **Val Loss**: `1.1620`
- **Val Macro F1**: `0.6667`
- **Test Accuracy**: `80.0%`
- **Saved Reports**:
  - Results JSON: [soft_results.json](file:///c:/Users/Jini%20John/projects/citcl/outputs/soft_results.json)
  - Text Report: [soft_classification_report.txt](file:///c:/Users/Jini%20John/projects/citcl/outputs/soft_classification_report.txt)
