import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
import pickle
from typing import Optional

# ================ BASE CLASS DEFINITION ================

class BaseClassifierDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data_dir = root_dir
        self.transform = transform
        self.classes = ['event', 'notevent']
        self.data = []
        self.labels = []
        self.num_classes = len(self.classes)

        print(f"[DEBUG] BaseClassifierDataset initialized with root_dir: {root_dir}")

        # Loop through each class directory and load file paths and labels
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            print(f"[DEBUG] Looking for class directory: {class_dir}")
            
            if not os.path.exists(class_dir):
                print(f"[ERROR] Class directory does not exist: {class_dir}")
                print(f"[DEBUG] Available directories in {root_dir}:")
                try:
                    for item in os.listdir(root_dir):
                        item_path = os.path.join(root_dir, item)
                        if os.path.isdir(item_path):
                            print(f"  - {item}/")
                        else:
                            print(f"  - {item}")
                except Exception as e:
                    print(f"  Error listing directory: {e}")
                continue
                
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.pkl'):
                    file_path = os.path.join(class_dir, file_name)
                    self.data.append(file_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx: int) -> tuple:
        """Retrieve a sample and its label by index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the sample (torch.Tensor) and its label (int).
        """
        file_path = self.data[idx]
        label = self.labels[idx]
        with open(file_path, 'rb') as f:
            sample = pickle.load(f)
            # (H,W, C) to (C,H,W) using np.transpose
            sample = np.transpose(sample, (2, 0, 1))
            
        
        if self.transform:
            sample = self.transform(sample)

        return sample, label




# ================ DATAMODULE DEFINITION ================

class ClassifierDataModule(LightningDataModule):
    def __init__(self, root_dir, batch_size=32, num_workers=15, transform=None, mode='semisplit'):
        super().__init__()
        self.data_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.mode = mode
        self.setup(mode=self.mode)
        

    def setup(self, mode: str = 'autosplit', stage: Optional[str] = None) -> None:
        """Set up the data loader with different splitting modes.
        
        Args:
            mode: The splitting mode ('autosplit', 'manualsplit', or 'semisplit').
            stage: Optional stage parameter for Lightning compatibility.
        """
        print(f"[DEBUG] ClassifierDataModule.setup() called with mode='{mode}', stage='{stage}', self.mode='{self.mode}'")
        
        # Use self.mode if mode parameter is the default
        if mode == 'autosplit' and hasattr(self, 'mode') and self.mode != 'autosplit':
            mode = self.mode
            print(f"[DEBUG] Using self.mode instead: '{mode}'")
            
        if mode == 'autosplit':
            # Load all data
            dataset = BaseClassifierDataset(self.data_dir, transform=self.transform)
            
            # Split data into train, val, and test sets
            train_size = int(0.7 * len(dataset))
            val_size = int(0.1 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset, [train_size, val_size, test_size])
        
        elif mode == 'manualsplit':
            # Rename according to the folder structure
            train_dataset = BaseClassifierDataset(os.path.join(self.data_dir, 'train'), transform=self.transform)
            self.train_dataset = train_dataset
            self.val_dataset = BaseClassifierDataset(os.path.join(self.data_dir, 'val'), transform=self.transform)
            self.test_dataset = BaseClassifierDataset(os.path.join(self.data_dir, 'test'), transform=self.transform)
            

            
        elif mode == 'semisplit':
            print(f"Setting up semisplit mode with root_dir: {self.data_dir}")
            train_val_dataset = BaseClassifierDataset(os.path.join(self.data_dir, 'TrainVal'), transform=self.transform)
            train_size = int(0.8 * len(train_val_dataset))
            val_size = len(train_val_dataset) - train_size
            print(f"TrainVal dataset size: {len(train_val_dataset)}, train_size: {train_size}, val_size: {val_size}")
            # train and val split
            self.train_dataset, self.val_dataset = random_split(
                train_val_dataset, [train_size, val_size])
            # test set
            self.test_dataset = BaseClassifierDataset(os.path.join(self.data_dir, 'Test'), transform=self.transform)
            print(f"Test dataset size: {len(self.test_dataset)}")
        
        else:
            raise ValueError(f'Unsupported mode: {mode}. Supported modes are: autosplit, manualsplit, semisplit')
        
        # Get num_classes from the original dataset or from any dataset
        if mode == 'autosplit':
            self.num_classes = dataset.num_classes
            sample, _ = dataset[0]
        elif mode == 'semisplit':
            self.num_classes = train_val_dataset.num_classes
            sample, _ = train_val_dataset[0]
        else:  # manualsplit
            self.num_classes = train_dataset.num_classes
            sample, _ = train_dataset[0]
        
        self.input_shape = sample.shape
        
        # Validate that setup completed successfully
        assert self.input_shape is not None, "input_shape must be set after setup"
        assert self.num_classes is not None, "num_classes must be set after setup"



    #  DataLoader definitions
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)