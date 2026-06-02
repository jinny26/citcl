import torch
import torch.nn as nn
from transformers import AutoModel

class SciBERTClassifier(nn.Module):
    """
    SciBERT citation intent classifier wrapping the base AutoModel.
    Supports freezing base model weights to enable fast classification head training.
    """
    def __init__(self, model_name="allenai/scibert_scivocab_uncased", num_classes=3, freeze_bert=False):
        super().__init__()
        print(f"Initializing SciBERTClassifier with base model: {model_name}")
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Determine hidden size
        hidden_size = self.bert.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Freeze SciBERT weights if specified
        if freeze_bert:
            print("Freezing SciBERT base weights (only training classification head).")
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            print("Fine-tuning full SciBERT model parameters.")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Forward pass through BERT base
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Use pooler output if available, otherwise use CLS token from last_hidden_state
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0]
            
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
