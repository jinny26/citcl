import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="SciBERT Citation Intent Classifier Training Configuration")
    
    # Model configuration
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="allenai/scibert_scivocab_uncased", 
        help="Hugging Face pre-trained model name or path"
    )
    parser.add_argument(
        "--freeze_bert", 
        action="store_true", 
        help="Freeze SciBERT base weights and only train the classification head"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        help="Training and evaluation batch size"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=2e-5, 
        help="Learning rate for optimization. Ideal is 2e-5 for full tuning, or 1e-3 if frozen."
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=128, 
        help="Maximum token sequence length (shorter lengths speed up CPU training)"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01, 
        help="Weight decay coefficient for AdamW"
    )
    
    # Execution options
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs", 
        help="Directory to save checkpoints, logs, and evaluation reports"
    )
    parser.add_argument(
        "--quick_test", 
        action="store_true", 
        help="Load only a small subset of the data (50 samples) and run for 1 epoch for quick validation"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=None, 
        help="Path to custom/local dataset files (especially relevant for the SOFT dataset)"
    )

    args = parser.parse_args()
    
    # Adjust defaults if quick test is enabled
    if args.quick_test:
        args.epochs = 1
        print(">>> Quick test enabled: setting epochs=1 and subsetting data to 50 samples.")
        
    # Adjust learning rate default if freezing SciBERT base
    if args.freeze_bert and args.lr == 2e-5:
        args.lr = 1e-3
        print(">>> SciBERT base is frozen: updating default learning rate to 1e-3 for the classification head.")
        
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args
