import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss



class Classifier(nn.Module):
    def __init__(self, input_dim, n_classes, dropout=0.2):
        super(Classifier, self).__init__()
        self.n_classes = n_classes
        self.embedder = None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, n_classes)
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.embedder(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if self.n_classes == 1:
            loss_function = MSELoss()
            loss = loss_function(logits.view(-1), labels.view(-1))
        else:
            loss_function = CrossEntropyLoss()
            loss = loss_function(logits.view(-1, self.n_classes), labels.view(-1))
        return logits, loss
