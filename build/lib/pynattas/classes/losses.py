import torch
from torch import nn
import torch.nn.functional as F

'''
class CategoricalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        log_softmax = nn.LogSoftmax(dim=1)
        log_probs = log_softmax(logits)
        
        print(f"targets dtype before conversion: {targets.dtype}")
        if not targets.dtype.is_integer:
            targets = targets.long()
        print(f"targets dtype after conversion: {targets.dtype}")
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=logits.shape[1])
        loss = -(targets_onehot * log_probs).sum(dim=1).mean()
        return loss
 

class CategoricalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        log_softmax = nn.LogSoftmax(dim=1)
        log_probs = log_softmax(logits)
        
        print(f"targets dtype before conversion: {targets.dtype}")
        if not torch.is_floating_point(targets):
            targets = targets.long()
            print(f"Shape of target: {targets.shape}")
        print(f"targets dtype after conversion: {targets.dtype}")
        
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=logits.shape[1])
        loss = -(targets_onehot * log_probs).sum(dim=1).mean()
        return loss 
'''
class CategoricalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # Print shapes for debugging
        #print(f"Logits shape: {logits.shape}")
        #print(f"Targets shape: {targets.shape}")

        # Ensure targets are in the correct shape and dtype
        if targets.ndim == 4:  # targets has shape (batch_size, num_classes, height, width)
            targets = torch.argmax(targets, dim=1)  # Convert one-hot to class indices
        
        #print(f"Converted Targets shape: {targets.shape}")

        # Compute the loss
        loss = self.criterion(logits, targets)
        
        return loss


class FocalLoss(nn.Module): #FocalLoss
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Convert targets from one-hot to class indices if necessary
        if targets.ndim == 4:
            targets = torch.argmax(targets, dim=1)

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        



    
