import os
import random
from typing import Optional, Tuple, List
from pathlib import Path
from datasets.RawVessels.loader import RawVesselsDataset

def train_test_split_simple(files_input: List[str], files_labels: List[str], 
                           test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Simple implementation of train_test_split without sklearn dependency.
    """
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
        dataset_path: Path to the dataset directory
        test_size: Proportion of dataset to include in the test split
        val_size: Proportion of remaining dataset to include in the validation split
        seed: Random state for reproducible splits
        is_3d: Whether the data is 3D (unused for now)
        despeckling: Despeckling method (unused for now)
        filter_size: Filter size for preprocessing (unused for now)
        resize_at: Resize dimension (unused for now)
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Ensure the dataset path exists
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
    
    # Load image and mask paths
    inputs_dir = dataset_path / 'inputs'
    masks_dir = dataset_path / 'masks'
    
    if not inputs_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Expected directories 'inputs' and 'masks' not found in {dataset_path}")
    
    # Get file paths
    input_files = sorted([str(x) for x in inputs_dir.glob('*.pkl')])
    label_files = sorted([str(x) for x in masks_dir.glob('*.pkl')])
    
    if len(input_files) == 0 or len(label_files) == 0:
        raise ValueError("No input or label files found in the dataset directory")
    
    if len(input_files) != len(label_files):
        raise ValueError(f"Mismatch between input files ({len(input_files)}) and label files ({len(label_files)})")
    
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