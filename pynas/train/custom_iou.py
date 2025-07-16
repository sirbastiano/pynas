import torch
    


def calculate_iou(logits, targets, num_classes=4):
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)  # Convert logits to class predictions
        targets = torch.argmax(targets, dim=1)  # Convert one-hot targets to class labels

        #print(f"Shape of logits: {logits.shape}")
        #print(f"Shape of preds: {preds.shape}")
        #print(f"Shape of targets: {targets.shape}")

        iou = []
        for cls in range(num_classes):
            pred_mask = (preds == cls)
            target_mask = (targets == cls)

            #print(f"Class {cls}:")
            #print(f"pred_mask shape: {pred_mask.shape}")
            #print(f"target_mask shape: {target_mask.shape}")

            intersection = (pred_mask & target_mask).float().sum((1, 2))
            union = (pred_mask | target_mask).float().sum((1, 2))

            #print(f"intersection: {intersection}")
            #print(f"union: {union}")

            iou.append((intersection + 1e-6) / (union + 1e-6))

        mean_iou = torch.stack(iou).mean(dim=0)  # Mean IoU across all classes
        #print(f"Mean IoU: {mean_iou}")
        return mean_iou
    
    
def calculate_iou_minclass(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Calculate IoU for the minority class (class 0).
    
    Args:
        logits (torch.Tensor): Model predictions with shape (batch_size, num_classes, height, width)
        targets (torch.Tensor): Ground truth targets with shape (batch_size, num_classes, height, width)
        num_classes (int): Number of classes, defaults to 4
        
    Returns:
        torch.Tensor: IoU values for the minority class (class 0) with shape (batch_size,)
    """
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)  # Convert logits to class predictions
        targets = torch.argmax(targets, dim=1)  # Convert one-hot targets to class labels

        # Calculate IoU only for class 0 (minority class)
        cls = 0
        pred_mask = (preds == cls)
        target_mask = (targets == cls)

        intersection = (pred_mask & target_mask).float().sum((1, 2))
        union = (pred_mask | target_mask).float().sum((1, 2))

        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou

