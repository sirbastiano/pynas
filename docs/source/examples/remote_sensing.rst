Remote Sensing Applications
==========================

This example demonstrates using PyNAS for remote sensing applications, including satellite imagery analysis, land cover classification, and change detection.

Overview
--------

Remote sensing applications have unique requirements:

- **Multi-spectral Data**: Handling different spectral bands
- **High Resolution**: Processing large images efficiently  
- **Temporal Analysis**: Analyzing time series data
- **Domain Adaptation**: Adapting to different sensors and regions

Satellite Image Classification
------------------------------

Land Cover Classification
~~~~~~~~~~~~~~~~~~~~~~~~~

Classify land cover types from Sentinel-2 imagery:

.. code-block:: python

    import torch
    import torch.nn as nn
    import numpy as np
    from pynas.core.population import Population
    from pynas.opt.evo import GeneticAlgorithm
    from pynas.blocks.convolutions import ConvBlock
    
    class RemoteSensingDataset(torch.utils.data.Dataset):
        """
        Dataset class for remote sensing imagery.
        """
        
        def __init__(self, images, labels, transform=None):
            """
            Initialize dataset.
            
            Args:
                images (numpy.ndarray): Satellite images [N, H, W, C]
                labels (numpy.ndarray): Land cover labels [N, H, W]
                transform: Data augmentation transforms
            """
            self.images = images
            self.labels = labels
            self.transform = transform
            self.num_classes = len(np.unique(labels))
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            
            # Convert to tensor
            image = torch.FloatTensor(image).permute(2, 0, 1)  # [C, H, W]
            label = torch.LongTensor(label)
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    class MultiSpectralBlock(nn.Module):
        """
        Custom block for multi-spectral data processing.
        """
        
        def __init__(self, in_channels, out_channels, num_bands=13):
            super().__init__()
            self.num_bands = num_bands
            
            # Band-specific processing
            self.band_conv = nn.ModuleList([
                nn.Conv2d(1, out_channels // num_bands, 3, padding=1)
                for _ in range(num_bands)
            ])
            
            # Cross-band fusion
            self.fusion_conv = nn.Conv2d(out_channels, out_channels, 1)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        
        def forward(self, x):
            # Process each band separately
            band_features = []
            for i, band_conv in enumerate(self.band_conv):
                band_input = x[:, i:i+1, :, :]  # Extract single band
                band_feat = band_conv(band_input)
                band_features.append(band_feat)
            
            # Concatenate band features
            fused = torch.cat(band_features, dim=1)
            
            # Apply fusion convolution
            output = self.fusion_conv(fused)
            output = self.relu(self.bn(output))
            
            return output
    
    def create_landcover_fitness_function(dataset_loader, device='cuda'):
        """
        Create fitness function for land cover classification.
        """
        
        def fitness_function(individual):
            model = individual.phenotype.to(device)
            model.eval()
            
            total_loss = 0
            correct_pixels = 0
            total_pixels = 0
            class_correct = torch.zeros(dataset_loader.dataset.num_classes)
            class_total = torch.zeros(dataset_loader.dataset.num_classes)
            
            criterion = nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for images, labels in dataset_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    
                    # Calculate pixel-wise accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct_pixels += (predicted == labels).sum().item()
                    total_pixels += labels.numel()
                    
                    # Per-class accuracy
                    for c in range(dataset_loader.dataset.num_classes):
                        class_mask = (labels == c)
                        class_correct[c] += (predicted[class_mask] == labels[class_mask]).sum().item()
                        class_total[c] += class_mask.sum().item()
            
            # Calculate metrics
            pixel_accuracy = correct_pixels / total_pixels
            mean_iou = calculate_mean_iou(class_correct, class_total)
            avg_loss = total_loss / len(dataset_loader)
            
            # Multi-objective fitness (accuracy and efficiency)
            model_size = sum(p.numel() for p in model.parameters())
            efficiency_score = 1.0 / (model_size / 1e6)  # Inverse of model size in millions
            
            # Combined fitness score
            fitness = 0.7 * pixel_accuracy + 0.2 * mean_iou + 0.1 * efficiency_score
            
            individual.fitness = fitness
            individual.metrics = {
                'pixel_accuracy': pixel_accuracy,
                'mean_iou': mean_iou,
                'loss': avg_loss,
                'model_size': model_size,
                'efficiency_score': efficiency_score
            }
            
            return fitness
        
        return fitness_function
    
    def calculate_mean_iou(class_correct, class_total):
        """Calculate mean Intersection over Union."""
        iou_scores = []
        for c in range(len(class_correct)):
            if class_total[c] > 0:
                iou = class_correct[c] / class_total[c]
                iou_scores.append(iou)
        return sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

Change Detection
~~~~~~~~~~~~~~~~

Detect changes between multi-temporal images:

.. code-block:: python

    class ChangeDetectionArchitecture(nn.Module):
        """
        Architecture for change detection in satellite imagery.
        """
        
        def __init__(self, num_bands=13, num_classes=2):
            super().__init__()
            self.num_bands = num_bands
            
            # Siamese encoder for temporal images
            self.encoder = self.build_encoder()
            
            # Change detection head
            self.change_head = nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, 1)
            )
        
        def build_encoder(self):
            """Build encoder for feature extraction."""
            layers = []
            in_ch = self.num_bands
            
            # Multi-scale feature extraction
            for out_ch in [64, 128, 256, 512]:
                layers.extend([
                    MultiSpectralBlock(in_ch, out_ch, self.num_bands),
                    nn.MaxPool2d(2, 2)
                ])
                in_ch = out_ch
            
            return nn.Sequential(*layers)
        
        def forward(self, x1, x2):
            """
            Forward pass for change detection.
            
            Args:
                x1: Image at time t1 [B, C, H, W]
                x2: Image at time t2 [B, C, H, W]
            """
            # Extract features from both images
            feat1 = self.encoder(x1)
            feat2 = self.encoder(x2)
            
            # Calculate feature difference
            diff = torch.abs(feat1 - feat2)
            
            # Detect changes
            change_map = self.change_head(diff)
            
            # Upsample to original resolution
            change_map = F.interpolate(
                change_map, size=x1.shape[-2:], 
                mode='bilinear', align_corners=False
            )
            
            return change_map
    
    def create_change_detection_fitness(dataset_loader, device='cuda'):
        """
        Create fitness function for change detection.
        """
        
        def fitness_function(individual):
            model = individual.phenotype.to(device)
            model.eval()
            
            total_loss = 0
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            criterion = nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for (img1, img2), labels in dataset_loader:
                    img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                    
                    outputs = model(img1, img2)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    
                    # Calculate change detection metrics
                    _, predicted = torch.max(outputs, 1)
                    
                    # Binary change detection metrics
                    changed_mask = (labels == 1)  # Changed pixels
                    unchanged_mask = (labels == 0)  # Unchanged pixels
                    
                    true_positives += (predicted[changed_mask] == 1).sum().item()
                    false_positives += (predicted[unchanged_mask] == 1).sum().item()
                    false_negatives += (predicted[changed_mask] == 0).sum().item()
            
            # Calculate F1 score
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            avg_loss = total_loss / len(dataset_loader)
            
            # Fitness combines F1 score and model efficiency
            model_size = sum(p.numel() for p in model.parameters())
            efficiency_score = 1.0 / (model_size / 1e6)
            
            fitness = 0.8 * f1_score + 0.2 * efficiency_score
            
            individual.fitness = fitness
            individual.metrics = {
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall,
                'loss': avg_loss,
                'model_size': model_size
            }
            
            return fitness
        
        return fitness_function

Hyperspectral Image Analysis
-----------------------------

Processing hyperspectral data with hundreds of bands:

.. code-block:: python

    class HyperspectralBlock(nn.Module):
        """
        Block for processing hyperspectral data.
        """
        
        def __init__(self, num_bands=224, out_channels=64):
            super().__init__()
            self.num_bands = num_bands
            
            # Spectral dimension reduction
            self.spectral_conv = nn.Conv1d(num_bands, out_channels, 1)
            
            # Spatial convolutions
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
            # Spectral-spatial attention
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, out_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels // 4, out_channels),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            # x shape: [B, H, W, Bands]
            B, H, W, Bands = x.shape
            
            # Reshape for spectral processing
            x_spectral = x.view(B * H * W, Bands, 1)
            x_spectral = self.spectral_conv(x_spectral)
            x_spectral = x_spectral.view(B, H, W, -1)
            
            # Convert to [B, C, H, W] for spatial processing
            x_spatial = x_spectral.permute(0, 3, 1, 2)
            
            # Apply spatial convolutions
            features = self.spatial_conv(x_spatial)
            
            # Apply attention
            attention_weights = self.attention(features)
            attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
            features = features * attention_weights
            
            return features
    
    def create_hyperspectral_architecture():
        """Create architecture for hyperspectral image classification."""
        
        class HyperspectralNet(nn.Module):
            def __init__(self, num_bands=224, num_classes=16):
                super().__init__()
                
                # Multi-scale hyperspectral processing
                self.block1 = HyperspectralBlock(num_bands, 64)
                self.block2 = HyperspectralBlock(64, 128)
                self.block3 = HyperspectralBlock(128, 256)
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes)
                )
            
            def forward(self, x):
                x = self.block1(x)
                x = F.max_pool2d(x, 2)
                x = self.block2(x)
                x = F.max_pool2d(x, 2)
                x = self.block3(x)
                x = self.classifier(x)
                return x
        
        return HyperspectralNet()

SAR Image Processing
--------------------

Process Synthetic Aperture Radar (SAR) imagery:

.. code-block:: python

    class SARProcessor(nn.Module):
        """
        Specialized processor for SAR imagery.
        """
        
        def __init__(self, in_channels=2, out_channels=64):  # I and Q channels
            super().__init__()
            
            # SAR-specific preprocessing
            self.complex_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            
            # Speckle reduction
            self.speckle_filter = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 5, padding=2, groups=out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
            # Edge enhancement
            self.edge_enhance = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            
        def forward(self, x):
            # Process complex SAR data
            complex_features = self.complex_conv(x)
            
            # Apply speckle reduction
            denoised = self.speckle_filter(complex_features)
            
            # Enhance edges
            enhanced = self.edge_enhance(denoised)
            
            return enhanced + complex_features  # Residual connection
    
    def create_sar_nas_config():
        """Create NAS configuration for SAR image analysis."""
        
        config = {
            'population_size': 25,
            'num_generations': 40,
            'mutation_rate': 0.15,
            'crossover_rate': 0.7,
            
            # SAR-specific architecture constraints
            'architecture_constraints': {
                'max_depth': 8,
                'min_channels': 32,
                'max_channels': 512,
                'use_sar_blocks': True,
                'speckle_reduction': True
            },
            
            # Training configuration
            'training': {
                'epochs': 50,
                'batch_size': 16,
                'learning_rate': 0.001,
                'optimizer': 'AdamW',
                'scheduler': 'CosineAnnealingLR'
            },
            
            # Evaluation metrics
            'metrics': ['accuracy', 'f1_score', 'iou', 'model_size']
        }
        
        return config

Complete Remote Sensing Example
-------------------------------

Here's a complete example for satellite image classification:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from pynas.core.population import Population
    from pynas.opt.evo import GeneticAlgorithm
    
    def run_remote_sensing_nas():
        """
        Complete remote sensing NAS example.
        """
        
        # Load remote sensing dataset
        print("Loading remote sensing dataset...")
        train_dataset = RemoteSensingDataset(
            images=np.load('sentinel2_images_train.npy'),
            labels=np.load('landcover_labels_train.npy')
        )
        
        val_dataset = RemoteSensingDataset(
            images=np.load('sentinel2_images_val.npy'),
            labels=np.load('landcover_labels_val.npy')
        )
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Define architecture space for remote sensing
        def create_rs_genome():
            """Create genome for remote sensing architecture."""
            genome = []
            
            # Encoder blocks
            channels = [64, 128, 256, 512]
            for i, ch in enumerate(channels):
                block = {
                    'type': np.random.choice(['multispectral', 'conv', 'resnet']),
                    'channels': ch,
                    'kernel_size': np.random.choice([3, 5, 7]),
                    'use_attention': np.random.choice([True, False]),
                    'dropout_rate': np.random.uniform(0.1, 0.5)
                }
                genome.append(block)
            
            # Decoder blocks for segmentation
            for i, ch in enumerate(reversed(channels[:-1])):
                block = {
                    'type': 'decoder',
                    'channels': ch,
                    'upsample_mode': np.random.choice(['bilinear', 'nearest']),
                    'skip_connection': True
                }
                genome.append(block)
            
            return genome
        
        # Create custom architecture builder
        class RemoteSensingBuilder:
            def __init__(self, num_bands=13, num_classes=10):
                self.num_bands = num_bands
                self.num_classes = num_classes
            
            def build_architecture(self, genome):
                """Build architecture from genome."""
                layers = []
                current_channels = self.num_bands
                
                # Build encoder
                encoder_features = []
                for gene in genome[:4]:  # First 4 are encoder blocks
                    if gene['type'] == 'multispectral':
                        layer = MultiSpectralBlock(current_channels, gene['channels'])
                    else:
                        layer = nn.Sequential(
                            nn.Conv2d(current_channels, gene['channels'], 
                                    gene['kernel_size'], padding=gene['kernel_size']//2),
                            nn.BatchNorm2d(gene['channels']),
                            nn.ReLU(inplace=True),
                            nn.Dropout(gene['dropout_rate'])
                        )
                    
                    layers.append(layer)
                    encoder_features.append(gene['channels'])
                    current_channels = gene['channels']
                    
                    # Add pooling
                    layers.append(nn.MaxPool2d(2, 2))
                
                # Build decoder
                for i, gene in enumerate(genome[4:]):  # Decoder blocks
                    target_channels = encoder_features[-(i+2)]
                    
                    # Upsampling
                    layers.append(nn.Upsample(scale_factor=2, mode=gene['upsample_mode']))
                    
                    # Decoder convolution
                    layers.append(nn.Conv2d(current_channels, target_channels, 3, padding=1))
                    layers.append(nn.BatchNorm2d(target_channels))
                    layers.append(nn.ReLU(inplace=True))
                    
                    current_channels = target_channels
                
                # Final classification layer
                layers.append(nn.Conv2d(current_channels, self.num_classes, 1))
                
                return nn.Sequential(*layers)
        
        # Initialize population
        builder = RemoteSensingBuilder(num_bands=13, num_classes=10)
        population = Population(
            population_size=20,
            genome_factory=create_rs_genome,
            architecture_builder=builder
        )
        
        # Create fitness function
        fitness_fn = create_landcover_fitness_function(val_loader)
        
        # Configure genetic algorithm
        ga = GeneticAlgorithm(
            population=population,
            fitness_function=fitness_fn,
            mutation_rate=0.1,
            crossover_rate=0.8,
            selection_method='tournament',
            tournament_size=3
        )
        
        # Run evolution
        print("Starting neural architecture search...")
        best_individuals = []
        
        for generation in range(30):
            print(f"Generation {generation + 1}/30")
            
            # Evaluate population
            ga.evaluate_population()
            
            # Get best individual
            best = ga.get_best_individual()
            best_individuals.append(best)
            
            print(f"Best fitness: {best.fitness:.4f}")
            print(f"Best metrics: {best.metrics}")
            
            # Evolve population
            if generation < 29:  # Don't evolve on last generation
                ga.evolve()
        
        # Return best architecture
        final_best = max(best_individuals, key=lambda x: x.fitness)
        print(f"\nFinal best architecture fitness: {final_best.fitness:.4f}")
        print(f"Final best metrics: {final_best.metrics}")
        
        return final_best
    
    if __name__ == "__main__":
        # Run remote sensing NAS
        best_architecture = run_remote_sensing_nas()
        
        # Save best model
        torch.save(best_architecture.phenotype.state_dict(), 'best_remote_sensing_model.pth')
        
        print("Remote sensing NAS completed successfully!")

Data Preprocessing Utilities
----------------------------

Utilities for handling remote sensing data:

.. code-block:: python

    class RemoteSensingPreprocessor:
        """
        Preprocessing utilities for remote sensing data.
        """
        
        def __init__(self):
            self.band_statistics = {}
        
        def normalize_bands(self, image, method='percentile'):
            """
            Normalize spectral bands.
            
            Args:
                image: Multi-band image [H, W, Bands]
                method: Normalization method ('percentile', 'minmax', 'zscore')
            """
            if method == 'percentile':
                # Normalize to 2nd and 98th percentiles
                normalized = np.zeros_like(image)
                for band in range(image.shape[-1]):
                    band_data = image[:, :, band]
                    p2, p98 = np.percentile(band_data, [2, 98])
                    normalized[:, :, band] = np.clip((band_data - p2) / (p98 - p2), 0, 1)
            
            elif method == 'minmax':
                # Min-max normalization
                normalized = (image - image.min()) / (image.max() - image.min())
            
            elif method == 'zscore':
                # Z-score normalization
                normalized = (image - image.mean()) / image.std()
            
            return normalized
        
        def calculate_indices(self, image, bands_config):
            """
            Calculate vegetation and other spectral indices.
            
            Args:
                image: Multi-band image
                bands_config: Dictionary mapping band names to indices
            """
            indices = {}
            
            # NDVI (Normalized Difference Vegetation Index)
            if 'red' in bands_config and 'nir' in bands_config:
                red = image[:, :, bands_config['red']]
                nir = image[:, :, bands_config['nir']]
                indices['ndvi'] = (nir - red) / (nir + red + 1e-8)
            
            # NDWI (Normalized Difference Water Index)
            if 'green' in bands_config and 'nir' in bands_config:
                green = image[:, :, bands_config['green']]
                nir = image[:, :, bands_config['nir']]
                indices['ndwi'] = (green - nir) / (green + nir + 1e-8)
            
            # EVI (Enhanced Vegetation Index)
            if all(band in bands_config for band in ['red', 'nir', 'blue']):
                red = image[:, :, bands_config['red']]
                nir = image[:, :, bands_config['nir']]
                blue = image[:, :, bands_config['blue']]
                indices['evi'] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
            
            return indices
        
        def create_composite_bands(self, image, indices):
            """
            Create composite image with original bands and calculated indices.
            """
            composite_bands = [image]
            
            for idx_name, idx_data in indices.items():
                composite_bands.append(idx_data[:, :, np.newaxis])
            
            composite = np.concatenate(composite_bands, axis=-1)
            return composite

Performance Optimization for Large Images
-----------------------------------------

Handle large satellite images efficiently:

.. code-block:: python

    class TiledInference:
        """
        Handle large images through tiled processing.
        """
        
        def __init__(self, model, tile_size=512, overlap=64):
            self.model = model
            self.tile_size = tile_size
            self.overlap = overlap
        
        def predict_large_image(self, large_image):
            """
            Predict on large image using tiled approach.
            
            Args:
                large_image: Large input image [H, W, C]
            
            Returns:
                Prediction map for entire image
            """
            H, W, C = large_image.shape
            
            # Calculate tile positions
            tiles = self.calculate_tile_positions(H, W)
            
            # Initialize output
            output = np.zeros((H, W), dtype=np.float32)
            weight_map = np.zeros((H, W), dtype=np.float32)
            
            for tile_info in tiles:
                # Extract tile
                tile = self.extract_tile(large_image, tile_info)
                
                # Predict on tile
                tile_tensor = torch.FloatTensor(tile).permute(2, 0, 1).unsqueeze(0)
                
                with torch.no_grad():
                    tile_pred = self.model(tile_tensor)
                    tile_pred = torch.softmax(tile_pred, dim=1)
                    tile_pred = tile_pred.squeeze(0).cpu().numpy()
                
                # Merge prediction
                self.merge_tile_prediction(output, weight_map, tile_pred, tile_info)
            
            # Normalize by weights
            output = output / (weight_map + 1e-8)
            
            return output
        
        def calculate_tile_positions(self, H, W):
            """Calculate tile positions with overlap."""
            tiles = []
            step = self.tile_size - self.overlap
            
            for y in range(0, H, step):
                for x in range(0, W, step):
                    # Ensure tile doesn't exceed image bounds
                    y_end = min(y + self.tile_size, H)
                    x_end = min(x + self.tile_size, W)
                    
                    # Adjust start position if needed
                    y_start = max(0, y_end - self.tile_size)
                    x_start = max(0, x_end - self.tile_size)
                    
                    tiles.append({
                        'y_start': y_start, 'y_end': y_end,
                        'x_start': x_start, 'x_end': x_end
                    })
            
            return tiles
        
        def extract_tile(self, image, tile_info):
            """Extract tile from image."""
            return image[
                tile_info['y_start']:tile_info['y_end'],
                tile_info['x_start']:tile_info['x_end']
            ]
        
        def merge_tile_prediction(self, output, weight_map, tile_pred, tile_info):
            """Merge tile prediction into output."""
            # Create weight mask (higher weight in center, lower at edges)
            tile_h = tile_info['y_end'] - tile_info['y_start']
            tile_w = tile_info['x_end'] - tile_info['x_start']
            
            weight_mask = self.create_weight_mask(tile_h, tile_w)
            
            # Add to output
            output[
                tile_info['y_start']:tile_info['y_end'],
                tile_info['x_start']:tile_info['x_end']
            ] += tile_pred[0] * weight_mask  # Assuming class 1 is target
            
            weight_map[
                tile_info['y_start']:tile_info['y_end'],
                tile_info['x_start']:tile_info['x_end']
            ] += weight_mask
        
        def create_weight_mask(self, h, w):
            """Create weight mask with higher weights in center."""
            y, x = np.ogrid[:h, :w]
            cy, cx = h // 2, w // 2
            
            # Distance from center
            dist_y = np.abs(y - cy) / cy
            dist_x = np.abs(x - cx) / cx
            
            # Weight decreases with distance from center
            weight = (1 - dist_y) * (1 - dist_x)
            return weight

.. seealso::
   
   - :doc:`vessel_detection` for maritime remote sensing applications
   - :doc:`edge_optimization` for optimizing models for satellite hardware
   - :doc:`../tutorials/custom_architectures` for creating domain-specific architectures
