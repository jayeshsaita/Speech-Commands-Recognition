"""data_loader.py provides method to load datasets and returns dataloaders."""
import torch
from torchvision import transforms, datasets
from utils import util


def get_data_loaders(train_path, valid_path, test_path, batch_size,
                     num_workers, train_transforms=None, valid_transforms=None,
                     test_transforms=None):
    """
    Return train, validation and test dataloaders.

    Parameters:
        train_path - path to training data folder
        valid_path - path to validation data folder
        test_path - path to test data folder
        batch_size - batch size to use for dataloaders
        num_workers - num_workers to use for dataloaders
        train_transforms - transformations to apply on training data
        valid_transforms - transformations to apply on validation data
        test_transforms - transformations to apply on test data

    """
    mean, std = util.get_mean_and_std(train_path, num_workers)
    # get_mean_and_std() returns numpy array of mean and std
    # passing numpy array to normalize changes datatype of torch tensor
    # to numpy becuase of subtraction and division
    # This causes error when getting data from dataloader
    # Hence, converting numpy array to torch tensors
    mean = torch.Tensor(mean)
    std = torch.Tensor(std)
    print(f'Mean:{mean}')
    print(f'Std:{std}')

    if train_transforms:
        train_transforms = train_transforms
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    if valid_transforms:
        valid_transforms = valid_transforms
    else:
        valid_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    if test_transforms:
        test_transforms = test_transforms
    else:
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    train_dataset = datasets.ImageFolder(
        root=train_path, transform=train_transforms)

    valid_dataset = datasets.ImageFolder(
        root=valid_path, transform=valid_transforms)

    test_dataset = datasets.ImageFolder(
        root=test_path, transform=test_transforms)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)

    valid_data_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)

    return [train_data_loader, valid_data_loader, test_data_loader]
