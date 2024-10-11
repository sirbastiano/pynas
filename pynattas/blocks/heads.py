import torch.nn as nn
from .utils import Dropout
import torch.nn.functional as F


class ClassificationHead(nn.Sequential):
    """
        Classification Head for Neural Networks.
        This module represents a classification head typically used at the end of a neural network. It consists of a
        linear layer, a ReLU activation, dropout for regularization, and a final linear layer that maps to the number
        of classes. This head is designed to be attached to the feature-extracting layers of a network to perform
        classification tasks.

        Args:
            input_size (int): The size of the input features.
            num_classes (int, optional): The number of classes for classification. Defaults to 2.

        The sequence of operations is as follows: Linear -> ReLU -> Dropout -> Linear.
    """
    def __init__(self, input_size, num_classes=2):
        super(ClassificationHead, self).__init__(
            #nn.Linear(input_size, 512),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            Dropout(p=0.4),
            #nn.Linear(512, num_classes)
            nn.Linear(256, num_classes)
        )



class SegmentationHead(nn.Module):
    def __init__(self, input_channels, num_classes, num_layers=3):
        super(SegmentationHead, self).__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(input_channels, num_classes, kernel_size=1))  # Final layer for classes
        
        self.segmentation_head = nn.Sequential(*layers)

    def forward(self, x):
        x = self.segmentation_head(x)
        if self.segmentation_head[-1].out_channels == 1:
            x = torch.sigmoid(x)  # For binary segmentation
        else:
            x = F.softmax(x, dim=1)  # For multi-class segmentation
        return x