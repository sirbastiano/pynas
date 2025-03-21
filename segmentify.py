import torch
import torch.nn as nn


class GenericDecoder(nn.Module):
    """A generic decoder that can work with any encoder by adapting to its output shape."""
    
    def __init__(self, in_channels, out_channels):
        super(GenericDecoder, self).__init__()
        
        # Progressive upsampling blocks
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels, 64, 
                                  kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 
                                  kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, 
                                  kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Final output layer
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)
    
    def forward(self, x):
        for block in self.decoder_blocks:
            x = block(x)
        return self.final_conv(x)





class ModelConstructor:
    """Constructs a segmentation model using any provided encoder."""

    def __init__(self, encoder, dm, verbose=False):
        # Validate dataset metadata (dm)
        if not hasattr(dm, "num_classes") or not hasattr(dm, "input_shape"):
            raise ValueError("dm must have 'num_classes' and 'input_shape' attributes.")

        self.encoder = encoder
        self.num_classes = dm.num_classes
        self.input_shape = dm.input_shape
        self.verbose = verbose

        # Validate encoder
        try:
            next(self.encoder.parameters())
        except StopIteration:
            raise ValueError("Encoder appears to have no parameters.")
        try:
            # Check if encoder has a forward method
            if not hasattr(self.encoder, 'forward'):
                raise AttributeError("Encoder must have a 'forward' method.")
        except Exception as e:
            raise ValueError("Provided encoder does not follow expected API.") from e

        # Ensure input shape is correct
        if not isinstance(self.input_shape, tuple):
            raise TypeError("input_shape must be a tuple.")
        if len(self.input_shape) == 3:
            self.input_shape = (1,) + self.input_shape
        elif len(self.input_shape) != 4:
            raise ValueError("input_shape must be of length 3 or 4.")

        # Build the decoder head
        self.decoder = self.build_decoder(input_shape=self.input_shape)

        # Complete model
        self.model = nn.Sequential(
            self.encoder,
            self.decoder
        )

        # Validate model with a dummy test
        self.valid_model = self.dummy_test()

    def build_decoder(self, input_shape):
        """Constructs the U-Net decoder using encoder's output shape."""
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

        # Check that the features tensor has at least 2 dimensions.
        if features.dim() < 2:
            raise ValueError("Encoded features should have at least 2 dimensions.")
        
        feature_channels = features.shape[1]  # Extract channels from encoder output
        return GenericDecoder(feature_channels, out_channels=self.num_classes)


    def dummy_test(self):
        """Runs a dummy input through the model to check output shape."""
        try:
            device = next(self.encoder.parameters()).device
            dummy = torch.randn(*self.input_shape).float().to(device)
            output = self.model(dummy)

            if self.verbose:
                print("Output shape from the model:", output.shape)

            if not isinstance(output, torch.Tensor):
                raise TypeError("Output of the model should be a torch.Tensor.")

            expected_shape = (dummy.shape[0], self.num_classes, dummy.shape[2], dummy.shape[3])
            if output.shape != expected_shape:
                raise ValueError(f"Expected output shape {expected_shape}, but got {output.shape}.")

            return True
        except Exception as e:
            if self.verbose:
                print("An error occurred during dummy_test:", e)
            return False

    def forward(self, x):
        """Performs forward pass."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        return self.model(x)