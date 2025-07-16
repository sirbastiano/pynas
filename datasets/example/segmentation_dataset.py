import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler

class SegmentationDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'TrainVal', transform=None):
        """
        Dataset for semantic segmentation.
        
        Args:
            root_dir (str): Directory with all the images and masks.
            split (str): One of ['TrainVal', 'Test'] to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Define class names and number of classes
        self.classes = ['Background', 'BurntArea', 'Cloud', 'Waterbodies']
        self.num_classes = len(self.classes)

        # Getting the list of image and mask paths
        self.image_paths = []
        self.mask_paths = []

        image_dir = os.path.join(self.root_dir, split, 'numpy_images')
        mask_dir = os.path.join(self.root_dir, split, 'numpy_masks')

        image_filenames = os.listdir(image_dir)
        mask_filenames = os.listdir(mask_dir)

        for image_filename in image_filenames:
            if image_filename.endswith('.npy'):
                self.image_paths.append(os.path.join(image_dir, image_filename))

        for mask_filename in mask_filenames:
            if mask_filename.endswith('.npy'):
                self.mask_paths.append(os.path.join(mask_dir, mask_filename))

        # Ensure both image_paths and mask_paths are sorted to match corresponding pairs
        self.image_paths.sort()
        self.mask_paths.sort()
        
        # Validate that we have matching pairs
        assert len(self.image_paths) == len(self.mask_paths), f'Mismatch: {len(self.image_paths)} images vs {len(self.mask_paths)} masks'

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = np.load(img_path)
        if image.shape[-1] == 7:  # Assuming 7 channels
            image = image.transpose(2, 0, 1)
        
        mask = np.load(mask_path)
        
        # Convert multi-channel mask to single-channel class indices
        if mask.ndim == 3 and mask.shape[-1] == 4:  # Multi-channel mask
            mask = mask.transpose(2, 0, 1)  # Change to (C, H, W)
            # Convert from one-hot encoding to class indices
            mask = np.argmax(mask, axis=0).astype(np.int64)
        elif mask.ndim == 3 and mask.shape[0] == 4:  # Already in (C, H, W) format
            # Convert from one-hot encoding to class indices
            mask = np.argmax(mask, axis=0).astype(np.int64)
        elif mask.ndim == 2:  # Already single-channel
            mask = mask.astype(np.int64)
        else:
            raise ValueError(f'Unexpected mask shape: {mask.shape}')
        
        # Ensure mask values are within valid range
        assert mask.min() >= 0 and mask.max() < self.num_classes, f'Invalid mask values: min={mask.min()}, max={mask.max()}, num_classes={self.num_classes}'
        
        # Convert to torch tensors
        image = torch.from_numpy(image.astype(np.float32))
        mask = torch.from_numpy(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


class SegmentationDataModule(LightningDataModule):
    def __init__(self, root_dir: str, batch_size: int = 8, num_workers: int = 1, 
                 transform=None, val_split: float = 0.3):
        """
        Lightning DataModule for segmentation datasets.
        
        Args:
            root_dir (str): Root directory of the dataset.
            batch_size (int): Batch size for dataloaders.
            num_workers (int): Number of workers for dataloaders.
            transform (callable, optional): Optional transform to be applied.
            val_split (float): Fraction of training data to use for validation.
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.val_split = val_split

        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage='fit'):
        # Load the full dataset
        full_dataset = SegmentationDataset(
            root_dir=self.root_dir,
            split='TrainVal',
            transform=self.transform,
        )

        # Store class info for external access
        self.class_names = full_dataset.classes
        self.num_classes = full_dataset.num_classes
        # Calculate shape of input data
        sample, _ = full_dataset[0]
        # print(sample.shape)
        self.input_shape = sample.shape

        # Calculate the number of samples for validation
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size

        # Split the dataset into train and validation
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        # Load test dataset
        if stage == 'test':
            self.test_dataset = SegmentationDataset(
                root_dir=self.root_dir,
                split='Test',
                transform=self.transform,
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