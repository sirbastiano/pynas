import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class RawClassifierDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data_dir = root_dir
        self.transform = transform
        self.classes = ['Land', 'Ocean', 'Ship']
        self.data = []
        self.labels = []
        self.num_classes = len(self.classes)

        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(class_dir, file_name)
                    self.data.append(file_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]
        sample = np.load(file_path)
                
        sample = np.stack((sample.real, sample.imag), axis=0)

        if self.transform:
            sample = self.transform(sample)

        return sample, label

class RawClassifierDataModule(LightningDataModule):
    def __init__(self, root_dir, batch_size=32, num_workers=4, transform=None):
        super().__init__()
        self.data_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.setup()

    def setup(self, stage=None):
        # Load all data
        dataset = RawClassifierDataset(self.data_dir, transform=self.transform)
        
        self.num_classes = dataset.num_classes
        # Calculate shape of input data
        sample, _ = dataset[0]
        # TODO: This assume every image of same shape
        self.input_shape = sample.shape
        
        # Split data into train, val, and test sets
        train_size = int(0.2 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)