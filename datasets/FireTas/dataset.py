import os
import zarr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule


class FireSegmentationDataset(Dataset):
    """
    Dataset class for the fire_dataset.zarr dataset.
    
    This dataset loads images and segmentation masks from a zarr store.
    Each sample is a directory containing an 'img' and a 'label' subdirectory.
    """
    def __init__(self, root_dir, split='test', transform=None):
        """
        Args:
            root_dir (string): Directory with the zarr dataset.
            split (string): One of ['test', 'trainval'] to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        
        # Get all sample directories (numbered folders)
        self.samples = sorted([d for d in os.listdir(self.root_dir) 
                              if os.path.isdir(os.path.join(self.root_dir, d))])
        
        # Load first sample to determine shape and classes
        if len(self.samples) > 0:
            first_sample = zarr.open(os.path.join(self.root_dir, self.samples[0]), mode='r')
            self.img_shape = first_sample['img'].shape
            if 'label' in first_sample:
                # Count unique values but ensure at least 2 classes for binary segmentation
                unique_values = np.unique(first_sample['label'])
                # If only background (0) is present, we still want to report 2 classes
                # because the dataset is meant to detect fire (class 1)
                if len(unique_values) == 1 and unique_values[0] == 0:
                    self.num_classes = 2  # Force binary classification (background/fire)
                else:
                    self.num_classes = len(unique_values)
                    
                print(f"Unique label values found: {unique_values}, num_classes set to: {self.num_classes}")
            else:
                self.num_classes = 2  # Default to binary (background/fire) if not specified
            
            # Define class names (customize these based on your dataset)
            self.classes = ['Background', 'Fire']
            if self.num_classes > 2:
                for i in range(2, self.num_classes):
                    self.classes.append(f'Class_{i}')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_path = os.path.join(self.root_dir, self.samples[idx])
        
        # Open the zarr group for this sample
        sample_zarr = zarr.open(sample_path, mode='r')
        
        # Read image and label as numpy arrays
        # Access arrays through their subgroups
        img_array = sample_zarr['img']
        label_array = sample_zarr['label']
        
        # Load the data - check if direct access or needs slicing
        try:
            image = img_array[:]
        except IndexError:
            # Handle case where array doesn't expect slicing
            image = np.array(img_array)
            
        try:
            label = label_array[:]
        except IndexError:
            # Handle case where array doesn't expect slicing
            label = np.array(label_array)
        
        # Convert numpy arrays to torch tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            # Handle label transform - usually only applied to image
        
        return image, label


class FireSegmentationDataModule(LightningDataModule):
    """
    Lightning Data Module for Fire Segmentation Dataset.
    
    This manages train/validation/test splits and creates appropriate 
    dataloaders for each.
    """
    def __init__(self, root_dir, batch_size=8, num_workers=4, transform=None, val_split=0.2):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.val_split = val_split
        
        # Call setup to initialize datasets
        self.setup()
    
    def prepare_data(self):
        # Nothing to download or prepare in this case
        pass
    
    def setup(self, stage=None):
        # Load the full trainval dataset
        trainval_dataset = FireSegmentationDataset(
            root_dir=self.root_dir,
            split='trainval',
            transform=self.transform
        )
        
        # Store dataset info
        self.class_names = trainval_dataset.classes
        self.num_classes = trainval_dataset.num_classes
        
        # Get a sample to determine input shape
        if len(trainval_dataset) > 0:
            sample_img, _ = trainval_dataset[0]
            self.input_shape = sample_img.shape
        
        # Split into train and validation sets
        val_size = int(len(trainval_dataset) * self.val_split)
        train_size = len(trainval_dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            trainval_dataset, 
            [train_size, val_size]
        )
        
        # Load test dataset
        self.test_dataset = FireSegmentationDataset(
            root_dir=self.root_dir,
            split='test',
            transform=self.transform
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# Example usage:
if __name__ == "__main__":
    # Path to the zarr dataset
    zarr_path = "/Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/data/fire_dataset.zarr"
    
    # Create the data module
    data_module = FireSegmentationDataModule(
        root_dir=zarr_path,
        batch_size=8,
        num_workers=4,
        transform=None  # Add transforms as needed
    )
    
    # Use the dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    # Print dataset info
    print(f"Number of training samples: {len(data_module.train_dataset)}")
    print(f"Number of validation samples: {len(data_module.val_dataset)}")
    print(f"Number of test samples: {len(data_module.test_dataset)}")
    print(f"Input shape: {data_module.input_shape}")
    print(f"Number of classes: {data_module.num_classes}")
    print(f"Class names: {data_module.class_names}")
    
    # Example of iterating through the dataloader
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images shape: {images.shape}, Labels shape: {labels.shape}")
        # Only print details for the first batch
        if batch_idx == 0:
            break