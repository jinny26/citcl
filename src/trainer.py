import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

def train_model(model, dataset, args, pipeline_name="citation_intent"):
    """
    Main training and evaluation loop using PyTorch.
    """
    print(f"\n===== Starting Pipeline: {pipeline_name} =====")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Create DataLoaders
    # Note: dataset is a DatasetDict with train, validation, test splits
    train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset["test"], batch_size=args.batch_size, shuffle=False)
    
    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Filter parameters to only update those requiring gradients (important when SciBERT is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Simple linear learning rate decay schedule
    num_training_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0, 
        end_factor=0.1, 
        total_iters=num_training_steps
    )
    
    best_val_f1 = -1.0
    best_model_path = os.path.join(args.output_dir, f"{pipeline_name}_best_model.pt")
    
    training_stats = []
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        epoch_start_time = time.time()
        
        # Training phase
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Time: {epoch_time:.1f}s")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val F1 (Macro): {val_metrics['macro_f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Save training statistics
        training_stats.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "time_seconds": epoch_time
        })
        
        # Check if validation F1 is the best so far
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            print(f"  New best model validation F1! Saving model checkpoint to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
            
    # Load best model checkpoint for final evaluation on test set
    if os.path.exists(best_model_path):
        print(f"Loading best checkpoint from {best_model_path} for final test evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        
    # Final evaluation on the Test set
    print("\nRunning final evaluation on the Test Set...")
    test_metrics = evaluate_model(model, test_loader, criterion, device, generate_report=True)
    
    print("\n===== Test Performance Summary =====")
    print(f"Loss:       {test_metrics['loss']:.4f}")
    print(f"Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"Macro F1:   {test_metrics['macro_f1']:.4f}")
    print(f"Micro F1:   {test_metrics['micro_f1']:.4f}")
    print("\nDetailed Classification Report:")
    print(test_metrics["classification_report"])
    
    # Save all pipeline outputs
    results_to_save = {
        "pipeline_name": pipeline_name,
        "config": {
            "model_name": args.model_name,
            "freeze_bert": args.freeze_bert,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "max_length": args.max_length,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "quick_test": args.quick_test
        },
        "training_stats": training_stats,
        "test_metrics": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "micro_f1": test_metrics["micro_f1"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"]
        }
    }
    
    results_path = os.path.join(args.output_dir, f"{pipeline_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=4)
    print(f"Results report successfully saved to {results_path}")
    
    report_path = os.path.join(args.output_dir, f"{pipeline_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(test_metrics["classification_report"])
    print(f"Classification report text saved to {report_path}")
    
    return test_metrics

def evaluate_model(model, data_loader, criterion, device, generate_report=False):
    """
    Evaluates the model on the given dataset loader and computes standard metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    
    # Calculate performance metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "precision": precision,
        "recall": recall
    }
    
    if generate_report:
        # Create standard scikit-learn classification report
        # We handle cases where some labels might be missing or out of bounds
        unique_labels = sorted(list(set(all_labels + all_preds)))
        target_names = [f"Class {i}" for i in unique_labels]
        metrics["classification_report"] = classification_report(
            all_labels, 
            all_preds, 
            labels=unique_labels,
            target_names=target_names, 
            zero_division=0
        )
        
    return metrics
