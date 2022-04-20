import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, EMNIST, FashionMNIST, CIFAR10


def get_transform(dataset: str = 'mnist'):
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return img_transform

def get_train_val_dataloaders(dataset: str = 'mnist', batch_size: int = 128, p_validation: float = 0.1, seed: int = 3,
                              return_img_dim: bool = True, return_alpha: bool = True):
    '''
    Returns a train dataloader and a validation dataloader if p_validation > 0.
    :param dataset: dataset to retrieve. Either "mnist", "emnist" or "fashionmnist".
    :param batch_size: batch size for each dataloader.
    :param p_validation: percentage of original training set to take for validation
    :param seed: manual seed for reproducibility
    :param return_img_dim: whether to return the img dimensions
    :return: train_dataloader, val_dataloader (if p_validation > 0), img_dim (if return_img_dim), alpha (if return_alpha)
    '''

    assert (p_validation >= 0) and (p_validation <= 1)

    img_transform = get_transform(dataset)

    if dataset.lower() == 'mnist':
        train_dataset = MNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
        alpha = 1e-6
    elif dataset.lower() == 'emnist':
        train_dataset = EMNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
        alpha = 1e-6
    elif dataset.lower() == 'fashionmnist':
        train_dataset = FashionMNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
        alpha = 1e-6
    elif dataset.lower() == 'cifar10' or dataset.lower() == 'cifar':
        train_dataset = CIFAR10(root='./data/MNIST', download=True, train=True, transform=img_transform)
        alpha = .05

    if p_validation > 0:
        torch_rng = torch.Generator().manual_seed(seed)
        size_validation = round(p_validation * len(train_dataset))
        size_train = len(train_dataset) - size_validation
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [size_train, size_validation],
                                                             generator=torch_rng)
        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        return_tuple = [train_dataloader, validation_dataloader]
    else:
        return_tuple = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True)]

    if return_img_dim:
        img_dim = train_dataset[0][0].shape
        return_tuple.append(img_dim)
    if return_alpha:
        return_tuple.append(alpha)

    return return_tuple

def get_test_dataloader(dataset: str = 'mnist', batch_size: int = 128):

    img_transform = get_transform(dataset)

    if dataset.lower() == 'mnist':
        test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)
    elif dataset.lower() == 'emnist':
        test_dataset = EMNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)
    elif dataset.lower() == 'fashionmnist':
        test_dataset = FashionMNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)
    elif dataset.lower() == 'cifar10' or dataset.lower() == 'cifar':
        test_dataset = CIFAR10(root='./data/MNIST', download=True, train=False, transform=img_transform)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader
