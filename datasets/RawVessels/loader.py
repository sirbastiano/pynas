import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from pathlib import Path
import random
from typing import Optional, Tuple, List, Union, Any

class RawVesselsDataset(Dataset):
    """Dataset for vessel segmentation tasks."""
    
    def __init__(self, image_paths: List[Union[str, Path]], mask_paths: List[Union[str, Path]], transform: Optional[Any] = None) -> None:
        """
        Initialize the RawVesselsDataset.
        
        Args:
            image_paths (List[Union[str, Path]]): List of paths to image files.
            mask_paths (List[Union[str, Path]]): List of paths to mask files.
            transform (Optional[Any]): Optional transform to be applied on samples.
            
        Raises:
            ValueError: If number of image and mask paths don't match or files don't end with .pkl.
            FileNotFoundError: If any of the specified paths don't exist.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
        # Define class names and number of classes
        self.classes = ['Background', 'Vessel']
        self.num_classes = len(self.classes)
        
        # Convert to Path objects if they're strings
        self.image_paths = [Path(p) if isinstance(p, str) else p for p in image_paths]
        self.mask_paths = [Path(p) if isinstance(p, str) else p for p in mask_paths]
        
        # Validation with proper assertions
        assert len(self.image_paths) == len(self.mask_paths), f'Number of image paths ({len(self.image_paths)}) must match number of mask paths ({len(self.mask_paths)})'
        
        missing_images = [p for p in self.image_paths if not p.exists()]
        assert not missing_images, f'Image paths do not exist: {missing_images}'
        
        missing_masks = [p for p in self.mask_paths if not p.exists()]
        assert not missing_masks, f'Mask paths do not exist: {missing_masks}'
        
        invalid_image_exts = [p for p in self.image_paths if not p.name.endswith('.pkl')]
        assert not invalid_image_exts, f'All image paths must end with .pkl: {invalid_image_exts}'
        
        invalid_mask_exts = [p for p in self.mask_paths if not p.name.endswith('.pkl')]
        assert not invalid_mask_exts, f'All mask paths must end with .pkl: {invalid_mask_exts}'
        
        self.image_dir = self.image_paths[0].parent
        self.mask_dir = self.mask_paths[0].parent

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of (image, mask) tensors.
                Image shape: (C, H, W) and Mask shape: (num_classes, H, W) in one-hot format.
            
        Raises:
            AssertionError: If mask values are not in valid range [0, 1].
            ValueError: If mask dimensions are unexpected.
        """
        assert 0 <= idx < len(self), f'Index {idx} out of range [0, {len(self)})'
        
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load the image and mask using pickle
        with open(img_path, 'rb') as f:
            image = pickle.load(f)
        with open(mask_path, 'rb') as f:
            mask = pickle.load(f)

        # Convert image to torch tensor with correct data type
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image, dtype=np.float32))
        else:
            image = image.float()

        # Convert mask to numpy first for easier processing
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.array(mask)

        # Handle different mask formats and convert to single-channel class indices first
        if mask_np.ndim == 3:
            if mask_np.shape[0] == 2:  # (C, H, W) format with 2 channels
                # Already in one-hot format, convert to class indices temporarily
                mask_np = np.argmax(mask_np, axis=0)
            elif mask_np.shape[-1] == 2:  # (H, W, C) format with 2 channels
                # Convert from one-hot to class indices
                mask_np = np.argmax(mask_np, axis=-1)
            elif mask_np.shape[0] == 1:  # Single channel in first dimension
                mask_np = mask_np[0]
            else:
                # If it's not a proper one-hot encoding, take the first channel
                mask_np = mask_np[0] if mask_np.shape[0] <= mask_np.shape[-1] else mask_np[:, :, 0]
        elif mask_np.ndim == 2:
            # Already 2D, keep as is for now
            pass
        else:
            raise ValueError(f'Unexpected mask dimensions: {mask_np.shape}')

        # Convert mask to binary format for vessel segmentation (class indices)
        # Ensure mask contains only 0 (background) and 1 (vessel)
        unique_values = np.unique(mask_np)
        
        if len(unique_values) == 1:
            # Only one unique value - check if it's valid
            if unique_values[0] not in [0, 1]:
                # If single value is not 0 or 1, convert to background
                mask_np = np.zeros_like(mask_np, dtype=np.int64)
            else:
                mask_np = mask_np.astype(np.int64)
        elif len(unique_values) == 2:
            # Two unique values - map to binary
            if set(unique_values) == {0, 1}:
                # Already binary
                mask_np = mask_np.astype(np.int64)
            else:
                # Map lowest value to 0, highest to 1
                mask_np = (mask_np == unique_values[1]).astype(np.int64)
        else:
            # More than 2 unique values - binarize based on threshold
            # Assume any non-zero value represents vessel (class 1)
            mask_np = (mask_np > 0).astype(np.int64)
        
        # Ensure mask values are only 0 and 1
        mask_np = np.clip(mask_np, 0, 1).astype(np.int64)
        
        # Convert class indices to one-hot encoding (C, H, W) format
        height, width = mask_np.shape
        mask_onehot = np.zeros((self.num_classes, height, width), dtype=np.float32)
        
        # Create one-hot encoding
        mask_onehot[0] = (mask_np == 0).astype(np.float32)  # Background channel
        mask_onehot[1] = (mask_np == 1).astype(np.float32)  # Vessel channel
        
        # Convert to torch tensor
        mask = torch.from_numpy(mask_onehot).float()
        
        # Final validation
        assert mask.shape[0] == self.num_classes, f'Mask should have {self.num_classes} channels, got {mask.shape[0]}'
        assert mask.ndim == 3, f'Mask must be 3D (C, H, W) after processing, got shape {mask.shape}'
        assert torch.all((mask == 0) | (mask == 1)), 'Mask values must be 0 or 1 in one-hot encoding'
        assert torch.allclose(mask.sum(dim=0), torch.ones(height, width)), 'Each pixel should belong to exactly one class'
        
        # Debug info - remove after confirming it works
        # print(f'Sample {idx}: Image shape: {image.shape}, Mask shape: {mask.shape}, Mask channels sum check: {mask.sum(dim=0).unique().tolist()}')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def train_test_split_simple(files_input: List[str], files_labels: List[str], 
                           test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Simple implementation of train_test_split without sklearn dependency.
    
    Args:
        files_input (List[str]): List of input file paths.
        files_labels (List[str]): List of label file paths.
        test_size (float): Proportion of dataset to include in test split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        Tuple[List[str], List[str], List[str], List[str]]: Train inputs, test inputs, train labels, test labels.
        
    Raises:
        AssertionError: If input and label lists have different lengths or test_size is invalid.
    """
    assert len(files_input) == len(files_labels), f'Input and label lists must have same length: {len(files_input)} vs {len(files_labels)}'
    assert 0.0 < test_size < 1.0, f'test_size must be between 0 and 1, got {test_size}'
    
    # Set random seed for reproducibility
    random.seed(random_state)
    
    # Create combined list with indices
    combined = list(zip(files_input, files_labels))
    
    # Shuffle the combined list
    random.shuffle(combined)
    
    # Calculate split index
    n_test = int(len(combined) * test_size)
    
    # Split the data
    test_data = combined[:n_test]
    train_data = combined[n_test:]
    
    # Unzip the data
    train_inputs, train_labels = zip(*train_data) if train_data else ([], [])
    test_inputs, test_labels = zip(*test_data) if test_data else ([], [])
    
    return list(train_inputs), list(test_inputs), list(train_labels), list(test_labels)


def get_real_data_sm(dataset_path: str, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42, 
                     is_3d: bool = False, despeckling: Optional[str] = None, filter_size: int = 7, 
                     resize_at: Optional[int] = None) -> Tuple[RawVesselsDataset, RawVesselsDataset, RawVesselsDataset]:
    """
    Split dataset into train, validation and test sets.
    
    Args:
        dataset_path (str): Path to the dataset directory.
        test_size (float): Proportion of dataset to include in the test split.
        val_size (float): Proportion of remaining dataset to include in the validation split.
        seed (int): Random state for reproducible splits.
        is_3d (bool): Whether the data is 3D (unused for now).
        despeckling (Optional[str]): Despeckling method (unused for now).
        filter_size (int): Filter size for preprocessing (unused for now).
        resize_at (Optional[int]): Resize dimension (unused for now).
    
    Returns:
        Tuple[RawVesselsDataset, RawVesselsDataset, RawVesselsDataset]: Train, validation, and test datasets.
        
    Raises:
        FileNotFoundError: If dataset path or required directories don't exist.
        ValueError: If no files found or file count mismatch.
        AssertionError: If validation parameters are invalid.
    """
    assert 0.0 < test_size < 1.0, f'test_size must be between 0 and 1, got {test_size}'
    assert 0.0 < val_size < 1.0, f'val_size must be between 0 and 1, got {val_size}'
    assert test_size + val_size < 1.0, f'test_size + val_size must be < 1.0, got {test_size + val_size}'
    
    # Ensure the dataset path exists
    dataset_path = Path(dataset_path)
    assert dataset_path.exists(), f'Dataset path {dataset_path} does not exist'
    
    # Load image and mask paths
    inputs_dir = dataset_path / 'inputs'
    masks_dir = dataset_path / 'masks'
    
    assert inputs_dir.exists() and masks_dir.exists(), f'Expected directories "inputs" and "masks" not found in {dataset_path}'
    
    # Get file paths
    input_files = sorted([str(x) for x in inputs_dir.glob('*.pkl')])
    label_files = sorted([str(x) for x in masks_dir.glob('*.pkl')])
    
    assert len(input_files) > 0 and len(label_files) > 0, 'No input or label files found in the dataset directory'
    assert len(input_files) == len(label_files), f'Mismatch between input files ({len(input_files)}) and label files ({len(label_files)})'
    
    # Split data into train, validation and test sets
    train_inputs, test_inputs, train_labels, test_labels = train_test_split_simple(
        input_files, label_files, test_size=test_size, random_state=seed
    )
    
    # Calculate validation size relative to remaining training data
    val_size_adjusted = val_size / (1 - test_size)
    train_inputs, val_inputs, train_labels, val_labels = train_test_split_simple(
        train_inputs, train_labels, test_size=val_size_adjusted, random_state=seed
    )
    
    # Create dataset objects
    train_dataset = RawVesselsDataset(train_inputs, train_labels, transform=None)
    val_dataset = RawVesselsDataset(val_inputs, val_labels, transform=None)
    test_dataset = RawVesselsDataset(test_inputs, test_labels, transform=None)
    
    return train_dataset, val_dataset, test_dataset


class RawVesselsDataModule(LightningDataModule):
    """Lightning DataModule for vessel segmentation datasets."""
    
    def __init__(self, root_dir: str, batch_size: int = 8, num_workers: int = 4, transform: Optional[Any] = None, 
                 test_size: float = 0.15, val_size: float = 0.15, seed: int = 42) -> None:
        """
        Initialize the RawVesselsDataModule.
        
        Args:
            root_dir (str): Root directory of the dataset.
            batch_size (int): Batch size for dataloaders.
            num_workers (int): Number of workers for dataloaders.
            transform (Optional[Any]): Optional transform to be applied.
            test_size (float): Fraction of data to use for testing.
            val_size (float): Fraction of data to use for validation.
            seed (int): Random seed for reproducible splits.
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = min(num_workers, os.cpu_count() or 1)  # Cap at available CPUs
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed

    def prepare_data(self) -> None:
        """Prepare data (download, etc.). Called only once per node."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for training, validation, and testing.
        
        Args:
            stage (Optional[str]): Stage identifier ('fit', 'test', or None).
        """
        # Use the get_real_data_sm function to get train, val, test datasets
        self.train_dataset, self.val_dataset, self.test_dataset = get_real_data_sm(
            dataset_path=self.root_dir,
            test_size=self.test_size,
            val_size=self.val_size,
            seed=self.seed,
            is_3d=False,
            despeckling=None,
            filter_size=7,
            resize_at=None
        )
        
        # Store class info for external access
        self.class_names = self.train_dataset.classes
        self.num_classes = self.train_dataset.num_classes
        
        # Calculate shape of input data
        if len(self.train_dataset) > 0:
            sample, _ = self.train_dataset[0]
            self.input_shape = sample.shape

    def train_dataloader(self) -> DataLoader:
        """
        Get training dataloader.
        
        Returns:
            DataLoader: Training dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """
        Get validation dataloader.
        
        Returns:
            DataLoader: Validation dataloader.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """
        Get test dataloader.
        
        Returns:
            DataLoader: Test dataloader.
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
