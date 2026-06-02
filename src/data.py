import os
import json
import urllib.request
import tarfile
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

# Mapping string labels to integers for ACL-ARC
ACL_ARC_LABEL_MAP = {
    "Background": 0,
    "Uses": 1,
    "CompareOrContrast": 2,
    "Extends": 3,
    "Motivation": 4,
    "FutureWork": 5
}

def preprocess_dataset(dataset, text_col, label_col, tokenizer, max_length, quick_test=False, label_map=None):
    """
    Standardizes text and label columns, tokenizes, and returns a dataset ready for training.
    """
    if quick_test:
        dataset = dataset.select(range(min(50, len(dataset))))
        
    def tokenize_function(examples):
        # Tokenize the text column
        tokenized = tokenizer(
            examples[text_col],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        
        # Convert labels to integer if needed
        labels = examples[label_col]
        if isinstance(labels, list):
            processed_labels = []
            for lbl in labels:
                if label_map and isinstance(lbl, str) and lbl in label_map:
                    processed_labels.append(label_map[lbl])
                elif isinstance(lbl, str) and label_map:
                    # Fallback if label is string but not in map (e.g., case-sensitive mismatch)
                    # Find closest match or default to 0
                    matched = False
                    for k, v in label_map.items():
                        if k.lower() == lbl.lower():
                            processed_labels.append(v)
                            matched = True
                            break
                    if not matched:
                        processed_labels.append(0)
                else:
                    processed_labels.append(int(lbl) if lbl is not None else 0)
            tokenized["label"] = processed_labels
        else:
            if label_map and isinstance(labels, str) and labels in label_map:
                tokenized["label"] = label_map[labels]
            else:
                tokenized["label"] = int(labels) if labels is not None else 0
                
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Ensure torch format
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized_dataset

def load_scicite_data(tokenizer, args):
    """
    Loads and tokenizes the SciCite dataset.
    Downloads the raw dataset from AI2 S3 bucket if not present locally, bypasses HF loading scripts.
    Label mapping: method -> 0, background -> 1, result -> 2
    """
    scicite_dir = os.path.join(args.output_dir, "scicite")
    train_file = os.path.join(scicite_dir, "scicite", "train.jsonl")
    val_file = os.path.join(scicite_dir, "scicite", "dev.jsonl")
    test_file = os.path.join(scicite_dir, "scicite", "test.jsonl")
    
    if not os.path.exists(train_file):
        print("Downloading SciCite dataset archive from S3...")
        os.makedirs(scicite_dir, exist_ok=True)
        tar_path = os.path.join(scicite_dir, "scicite.tar.gz")
        url = "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz"
        try:
            urllib.request.urlretrieve(url, tar_path)
            print("Extracting SciCite dataset archive...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=scicite_dir)
            print("Extraction completed successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to download/extract SciCite dataset from {url}: {e}")
            
    print(f"Loading SciCite dataset from local JSONL files in {scicite_dir}...")
    raw_dataset = load_dataset("json", data_files={
        "train": train_file,
        "validation": val_file,
        "test": test_file
    })
    
    train_dataset = preprocess_dataset(
        raw_dataset["train"], 
        text_col="string", 
        label_col="label", 
        tokenizer=tokenizer, 
        max_length=args.max_length, 
        quick_test=args.quick_test
    )
    
    val_dataset = preprocess_dataset(
        raw_dataset["validation"], 
        text_col="string", 
        label_col="label", 
        tokenizer=tokenizer, 
        max_length=args.max_length, 
        quick_test=args.quick_test
    )
    
    test_dataset = preprocess_dataset(
        raw_dataset["test"], 
        text_col="string", 
        label_col="label", 
        tokenizer=tokenizer, 
        max_length=args.max_length, 
        quick_test=args.quick_test
    )
    
    num_classes = 3
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    }), num_classes

def load_acl_arc_data(tokenizer, args):
    """
    Loads and tokenizes the ACL-ARC dataset from Hugging Face (using a script-free repository).
    """
    print("Loading ACL-ARC dataset from Hugging Face...")
    raw_dataset = load_dataset("hrithikpiyush/acl-arc")
    
    train_dataset = preprocess_dataset(
        raw_dataset["train"], 
        text_col="text", 
        label_col="intent", 
        tokenizer=tokenizer, 
        max_length=args.max_length, 
        quick_test=args.quick_test
    )
    
    val_dataset = preprocess_dataset(
        raw_dataset["validation"], 
        text_col="text", 
        label_col="intent", 
        tokenizer=tokenizer, 
        max_length=args.max_length, 
        quick_test=args.quick_test
    )
    
    test_dataset = preprocess_dataset(
        raw_dataset["test"], 
        text_col="text", 
        label_col="intent", 
        tokenizer=tokenizer, 
        max_length=args.max_length, 
        quick_test=args.quick_test
    )
    
    num_classes = len(ACL_ARC_LABEL_MAP)
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    }), num_classes

def load_soft_data(tokenizer, args):
    """
    Loads and tokenizes the SOFT framework dataset.
    If args.data_path is provided, loads from local JSON/CSV file.
    Otherwise, attempts to download the re-annotated SOFT ACL-ARC dataset from github or fallback to dummy.
    """
    # 1. Check if user specified a local path
    data_file_path = args.data_path
    
    if data_file_path and os.path.exists(data_file_path):
        print(f"Loading SOFT dataset from local file: {data_file_path}")
        if data_file_path.endswith('.csv'):
            df = pd.read_csv(data_file_path)
        else:
            # Assume JSON / JSONL
            try:
                df = pd.read_json(data_file_path)
            except ValueError:
                # Try reading line-by-line JSONL
                with open(data_file_path, 'r', encoding='utf-8') as f:
                    lines = [json.loads(line) for line in f]
                df = pd.DataFrame(lines)
    else:
        # 2. Try download from GitHub zhiyintan/SOFT
        # Fallback raw files from zhiyintan/SOFT repository
        soft_url = "https://raw.githubusercontent.com/zhiyintan/SOFT/main/dataset/SOFT_ACL-ARC.json"
        local_fallback = os.path.join(args.output_dir, "SOFT_ACL-ARC.json")
        
        if not os.path.exists(local_fallback):
            print(f"Attempting to download SOFT dataset from GitHub: {soft_url}")
            try:
                os.makedirs(args.output_dir, exist_ok=True)
                urllib.request.urlretrieve(soft_url, local_fallback)
                print("Download successful!")
            except Exception as e:
                print(f"Failed to download from GitHub: {e}")
                
        if os.path.exists(local_fallback):
            print(f"Loading SOFT dataset from downloaded file: {local_fallback}")
            try:
                # Read SOFT JSON format
                with open(local_fallback, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert the dictionary or list structure to DataFrame
                # In SOFT, it typically lists citation contexts with labels
                if isinstance(data, dict):
                    # If it's a dict of lists or structured by keys
                    records = []
                    for key, val in data.items():
                        if isinstance(val, dict):
                            val['id'] = key
                            records.append(val)
                        else:
                            records = data
                            break
                    if records:
                        df = pd.DataFrame(records)
                    else:
                        df = pd.DataFrame.from_dict(data, orient='index')
                else:
                    df = pd.DataFrame(data)
            except Exception as e:
                print(f"Error parsing downloaded JSON: {e}. Falling back to dummy dataset.")
                df = None
        else:
            print("SOFT dataset not found locally and download failed. Generating dummy dataset for sanity checks.")
            df = None
            
    # 3. Create dummy dataset if file loading/downloading failed completely (e.g. no internet and no file)
    if df is None or len(df) == 0:
        print("Creating dummy SOFT dataset for running sanity check pipelines...")
        dummy_data = {
            "text": [
                "We base our architecture on the standard transformer model.",
                "Prior work (Smith et al., 2020) suggested that SciBERT is a strong baseline.",
                "Our results outperform previous work in all tasks.",
                "We compare our model with standard classification pipelines.",
                "Future research should address the computational cost on CPU."
            ] * 20,
            "intent_label": [
                "Uses",
                "Background",
                "CompareOrContrast",
                "CompareOrContrast",
                "FutureWork"
            ] * 20
        }
        df = pd.DataFrame(dummy_data)
        
    # Standardize columns for SOFT.
    # Check what columns exist. We need text and label.
    text_col = "text"
    if "context" in df.columns:
        text_col = "context"
    elif "string" in df.columns:
        text_col = "string"
        
    label_col = "intent_label"
    if "label" in df.columns:
        label_col = "label"
    elif "intent" in df.columns:
        label_col = "intent"
    elif "intent_label_coarse" in df.columns:
        label_col = "intent_label_coarse"
        
    print(f"SOFT Dataset: Using text column '{text_col}' and label column '{label_col}'")
    
    # Get unique labels and map them
    unique_labels = sorted(df[label_col].unique())
    soft_label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    num_classes = len(soft_label_map)
    print(f"Detected {num_classes} classes in SOFT dataset: {soft_label_map}")
    
    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(df)
    
    # Split into train/validation/test (e.g. 80% train, 10% val, 10% test)
    # Using seed for reproducibility
    split_dataset = hf_dataset.train_test_split(test_size=0.2, seed=args.seed)
    train_val_split = split_dataset["train"].train_test_split(test_size=0.125, seed=args.seed) # 0.125 * 0.8 = 0.1 (10% overall)
    
    train_dataset = preprocess_dataset(
        train_val_split["train"],
        text_col=text_col,
        label_col=label_col,
        tokenizer=tokenizer,
        max_length=args.max_length,
        quick_test=args.quick_test,
        label_map=soft_label_map
    )
    
    val_dataset = preprocess_dataset(
        train_val_split["test"],
        text_col=text_col,
        label_col=label_col,
        tokenizer=tokenizer,
        max_length=args.max_length,
        quick_test=args.quick_test,
        label_map=soft_label_map
    )
    
    test_dataset = preprocess_dataset(
        split_dataset["test"],
        text_col=text_col,
        label_col=label_col,
        tokenizer=tokenizer,
        max_length=args.max_length,
        quick_test=args.quick_test,
        label_map=soft_label_map
    )
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    }), num_classes
