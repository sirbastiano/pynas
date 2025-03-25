import torch.nn as nn
import torch.nn.functional as F
import torch

class Dropout(nn.Module):
    """
    Dropout layer for regularization in neural networks.
    
    Args:
        p (float, optional): Probability of an element to be zeroed. Default: 0.5
        inplace (bool, optional): If set to True, will do this operation in-place. Default: False
    """
    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        self.p = p
        self.inplace = inplace
    
    def forward(self, x):
        return F.dropout(x, self.p, self.training, self.inplace)


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
    

class Classifier:
    def __init__(self, encoder, dm, verbose=False):
        # Validate that dm has the necessary attributes.
        if not hasattr(dm, "num_classes") or not hasattr(dm, "input_shape"):
            raise ValueError("dm must have 'num_classes' and 'input_shape' attributes.")
        
        self.encoder = encoder
        self.num_classes = dm.num_classes
        self.input_shape = dm.input_shape
        self.verbose = verbose
        if self.verbose:
            print(f"Input shape: {self.input_shape}")
        
        # Verify that encoder has parameters.
        try:
            next(self.encoder.parameters())
        except StopIteration:
            raise ValueError("Encoder appears to have no parameters.")
        except Exception as e:
            raise ValueError("Provided encoder does not follow expected API.") from e

        # Validate input_shape is a tuple and properly dimensioned.
        if not isinstance(self.input_shape, tuple):
            raise TypeError("input_shape must be a tuple.")
        if len(self.input_shape) == 3:
            if self.verbose:
                print("Adding channel dimension to input shape.")
                print(f"Original input shape: {self.input_shape}")
            self.input_shape = (1,) + self.input_shape
            if self.verbose:
                print(f"Updated input shape: {self.input_shape}")
        elif len(self.input_shape) != 4:
            raise ValueError("input_shape must be of length 3 or 4.")

        self.head_layer = self.build_head(input_shape=self.input_shape)
        
        self.model = nn.Sequential(
            self.encoder,
            self.head_layer
        )
        
        self.valid_model = self.dummy_test()

    def build_head(self, input_shape=(1, 2, 256, 256)):
        # Get the device from the encoder's parameters.
        try:
            device = next(self.encoder.parameters()).device
        except Exception as e:
            raise ValueError("Unable to determine device from encoder parameters.") from e
        
        # Run a dummy input through the encoder to get the feature shape.
        dummy = torch.randn(*input_shape).float().to(device)
        try:
            features = self.encoder(dummy)
        except Exception as e:
            raise RuntimeError("Error when running dummy input through encoder.") from e
        
        if not isinstance(features, torch.Tensor):
            raise TypeError("Encoder output should be a torch.Tensor.")

        if self.verbose:
            print("Feature map shape from the feature extractor:", features.shape)

        # Check that the features tensor has at least 2 dimensions.
        if features.dim() < 2:
            raise ValueError("Encoded features should have at least 2 dimensions.")
        
        # Determine the number of channels from the dummy output.
        feature_channels = features.shape[1]

        # Build the head layer.
        head_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_channels, self.num_classes)
        )
        if self.verbose:
            print("Constructed head layer:", head_layer)
        return head_layer
    
    
    def dummy_test(self):
        try:
            device = next(self.encoder.parameters()).device
            dummy = torch.randn(*self.input_shape).float().to(device)
            output = self.model(dummy)
            if self.verbose:
                print("Network test passed. Output shape from the model:", output.shape)
            
            if not isinstance(output, torch.Tensor):
                raise TypeError("Output of the model should be a torch.Tensor.")
            
            if output.shape[0] != dummy.shape[0]:
                raise ValueError("Batch size mismatch between input and output.")
            
            return True
        except Exception as e:
            if self.verbose:
                print("An error occurred during dummy_test:", e)
            return False
    
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        return self.model(x)