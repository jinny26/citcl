# Running SciCite Training on Google Colab (GitHub Workflow)

This document contains copy-pasteable cells to set up and run the SciCite training pipeline on Google Colab by cloning your code directly from GitHub.

---

### Step 1: Set Google Colab Runtime to GPU
Before running any cells, make sure your notebook is using GPU acceleration:
1. Go to the menu at the top: **Runtime** > **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (or any other available GPU).
3. Click **Save**.

---

### Step 2: Copy-Paste the Notebook Cells

Create new code cells in your Colab notebook and run them in sequence:

#### Cell 1: Clone the Repository and Change Directory
This clones your repository from GitHub and changes Colab's active directory to the project root.
```python
# Replace YOUR_GITHUB_USERNAME and citcl with your repository details
!git clone https://github.com/YOUR_GITHUB_USERNAME/citcl.git

# Change directory to the repository root
%cd citcl
```

#### Cell 2: Install Required Packages
Installs all standard dependencies (Hugging Face Transformers, Datasets, PyTorch, Scikit-learn):
```python
!pip install -r requirements.txt
```

#### Cell 3: Train SciCite Model (Full Fine-tuning on GPU)
This runs the full fine-tuning pipeline on the SciCite dataset. It will train for 3 epochs with a sequence length of 128.
```python
!python pipelines/train_scicite.py --epochs 3 --batch_size 16 --max_length 128
```

#### Cell 4: Visualize the Results
Once training finishes, you can load and display the final performance metrics:
```python
import json
import pandas as pd

# Load final results JSON
with open("outputs/scicite_results.json", "r") as f:
    results = json.load(f)

# Display test metrics
print("=== Final Test Set Metrics ===")
for metric, score in results["test_metrics"].items():
    print(f"{metric.capitalize()}: {score:.4f}")

# Display training stats per epoch
print("\n=== Training History ===")
history_df = pd.DataFrame(results["training_stats"])
display(history_df)

# Print final classification report
print("\n=== Classification Report ===")
with open("outputs/scicite_classification_report.txt", "r") as f:
    print(f.read())
```

---

### Step 3: Download Outputs
You can download the best model weights (`outputs/scicite_best_model.pt`) and reports directly from the file explorer tab on the left-hand panel of Google Colab (Right-click > **Download**).
