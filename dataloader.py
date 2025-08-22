import torch 
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision
import os
from pathlib import Path
from typing import Optional

def create_dataloaders(train_path:str|Path,
                       test_path: str|Path,
                       transform_data: torchvision.transforms.Compose,
                       valid_path: str|Path,
                       batch_size:int=32,
                       num_workers:int=os.cpu_count() or 1):
    """ 
    Create DataLoaders for train, validation (optional), and test sets.
    
    Args:
        train_path: Path to training dataset (with class subfolders).
        test_path: Path to test dataset (with class subfolders).
        valid_path: Optional path to validation dataset.
        transform_data: torchvision.transforms.Compose to apply to images.
        batch_size: Number of samples per batch.
        num_workers: Number of workers for DataLoader.
    
    Returns:
        train_dataloader, valid_dataloader, test_dataloader, class_names
    """
    train_data = ImageFolder(root=train_path,
                             transform=transform_data)
    valid_data = ImageFolder(root=valid_path,
                            transform=transform_data)
    
    test_data = ImageFolder(root=test_path,
                            transform=transform_data)
    class_names = train_data.classes
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_data,
                                    num_workers=num_workers,
                                    batch_size=batch_size,
                                    shuffle=False)
    test_dataloader = DataLoader(dataset=test_data,
                                 num_workers=num_workers,
                                 batch_size=batch_size,
                                 shuffle=False)
    
    return train_dataloader,valid_dataloader,test_dataloader,class_names

