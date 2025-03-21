import torch 
import torch.nn as nn


class ModelConstructor:
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