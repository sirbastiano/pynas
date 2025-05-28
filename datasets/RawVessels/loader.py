import os
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from pathlib import Path

class RawVesselsDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        """
        Args:
            image_paths (list): List of paths to image files
            mask_paths (list): List of paths to mask files
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
        # Define class names and number of classes
        self.classes = ['Background', 'Vessel',]
        self.num_classes = len(self.classes)
        
        
        
        #    
        if len(image_paths) != len(mask_paths):
            raise ValueError("The number of image paths must match the number of mask paths.")
        if not all(os.path.exists(p) for p in image_paths):
            raise FileNotFoundError("One or more image paths do not exist.")
        if not all(os.path.exists(p) for p in mask_paths):
            raise FileNotFoundError("One or more mask paths do not exist.")
        if not all(p.name.endswith('.pkl') for p in image_paths):
            raise ValueError("All image paths must end with '.pkl'.")
        if not all(p.name.endswith('.pkl') for p in mask_paths):
            raise ValueError("All mask paths must end with '.pkl'.")
        #
        self.image_dir = Path(image_paths[0]).parent
        self.mask_dir = Path(mask_paths[0]).parent
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Get the corresponding mask path
        img_path_obj = Path(img_path)
        mask_path = img_path_obj.parent.parent / 'masks' / img_path_obj.name
        mask_path = str(mask_path).replace('_L0_', '_L1_')  # Adjust the mask path if necessary
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file {mask_path} does not exist.")

        # Load the image and mask using pickle
        with open(img_path, 'rb') as f:
            image = pickle.load(f)
        with open(mask_path, 'rb') as f:
            mask = pickle.load(f)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


class RawVesselsDataModule(LightningDataModule):
    def __init__(self, root_dir, batch_size=8, num_workers=1, transform=None, train_split=0.7, val_split=0.15, test_split=0.15):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Split ratios must sum to 1"

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Find all image and mask files
        inputs_dir = os.path.join(self.root_dir, 'inputs')
        masks_dir = os.path.join(self.root_dir, 'masks')
        
        if not os.path.exists(inputs_dir) or not os.path.exists(masks_dir):
            raise FileNotFoundError(f"Cannot find inputs or masks directory in {self.root_dir}")
        
        image_paths = [os.path.join(inputs_dir, f) for f in os.listdir(inputs_dir) if f.endswith('.pkl')]
        mask_paths = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.pkl')]
        
        # Ensure both lists are sorted to match corresponding pairs
        image_paths.sort()
        mask_paths.sort()
        
        # Create a dataset with all samples
        full_dataset = RawVesselsDataset(image_paths, mask_paths, self.transform)
        
        # Calculate sizes for each split
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * self.train_split)
        val_size = int(dataset_size * self.val_split)
        test_size = dataset_size - train_size - val_size
        
        # Split the dataset into train, validation and test
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
