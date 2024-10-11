import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler

'''
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='TrainVal', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and masks.
            split (string): One of ['TrainVal', 'Test'] to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = np.load(img_path)
        mask = np.load(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

class SegmentationDataModule(LightningDataModule):
    def __init__(self, root_dir, batch_size=8, num_workers=1, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform  # No default transformation
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainval_dataset = SegmentationDataset(
                root_dir=self.root_dir,
                split='TrainVal',
                transform=self.transform,
            )
            self.test_dataset = SegmentationDataset(
                root_dir=self.root_dir,
                split='Test',
                transform=self.transform,
            )

    def train_dataloader(self):
        if self.trainer.num_nodes > 1:
            sampler = DistributedSampler(self.trainval_dataset, num_replicas=self.trainer.num_nodes, rank=self.trainer.global_rank)
            return DataLoader(
                dataset=self.trainval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler
            )
        else:
            return DataLoader(
                dataset=self.trainval_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )

    def val_dataloader(self):
        if self.trainer.num_nodes > 1:
            sampler = DistributedSampler(self.trainval_dataset, num_replicas=self.trainer.num_nodes, rank=self.trainer.global_rank)
            return DataLoader(
                dataset=self.trainval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler
            )
        else:
            return DataLoader(
                dataset=self.trainval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )

    def test_dataloader(self):
        if self.trainer.num_nodes > 1:
            sampler = DistributedSampler(self.test_dataset, num_replicas=self.trainer.num_nodes, rank=self.trainer.global_rank)
            return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler
            )
        else:
            return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )

    def predict_dataloader(self):
        pass
'''


#Segmentation module:
    

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='TrainVal', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and masks.
            split (string): One of ['TrainVal', 'Test'] to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = np.load(img_path)
        #print(f"Image shape in datasets.py {image.shape}")
        mask = np.load(mask_path)
        #print(f"Mask unique elements: {np.unique(mask)}, shape: {mask.shape}")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

'''
class SegmentationDataModule(LightningDataModule):
    def __init__(self, root_dir, batch_size=8, num_workers=1, transform=None, val_split=0.3):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform  # No default transformation
        self.num_workers = num_workers
        self.val_split = val_split

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainval_dataset = SegmentationDataset(
                root_dir=self.root_dir,
                split='TrainVal',
                transform=self.transform,
            )
            self.test_dataset = SegmentationDataset(
                root_dir=self.root_dir,
                split='Test',
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainval_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.trainval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        pass
'''

class SegmentationDataModule(LightningDataModule):
    def __init__(self, root_dir, batch_size=8, num_workers=1, transform=None, val_split=0.3):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform  # No default transformation
        self.num_workers = num_workers
        self.val_split = val_split  # Proportion of data to use for validation

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Load the full dataset
        full_dataset = SegmentationDataset(
            root_dir=self.root_dir,
            split='TrainVal',
            transform=self.transform,
        )

        # Calculate the number of samples for validation
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size

        # Split the dataset into train and validation
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
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


