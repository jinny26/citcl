import sys
import os
import torch
import numpy as np
import random
from transformers import AutoTokenizer

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import parse_args
from src.data import load_acl_arc_data
from src.model import SciBERTClassifier
from src.trainer import train_model

def main():
    args = parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print(f"Initializing ACL-ARC training pipeline...")
    print(f"Base model: {args.model_name}")
    print(f"Freeze SciBERT base: {args.freeze_bert}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load data
    dataset, num_classes = load_acl_arc_data(tokenizer, args)
    print(f"Dataset splits: {dataset.keys()}")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")
    print(f"Number of classes: {num_classes}")
    
    # Load model
    model = SciBERTClassifier(
        model_name=args.model_name,
        num_classes=num_classes,
        freeze_bert=args.freeze_bert
    )
    
    # Run training and evaluation pipeline
    train_model(model, dataset, args, pipeline_name="acl_arc")

if __name__ == "__main__":
    main()
