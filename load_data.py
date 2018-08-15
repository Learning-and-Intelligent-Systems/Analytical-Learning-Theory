'''
Created on 21 Nov 2017

@author: vermav1
'''
import torch
from torchvision import datasets, transforms
from affine_transforms import Rotation, Zoom



def load_mnist(data_aug, batch_size, test_batch_size,cuda, data_target_dir):

    if data_aug == 1:
        hw_size = 24
        transform_train = transforms.Compose([
                            transforms.RandomCrop(hw_size),                
                            transforms.ToTensor(),
                            Rotation(15),                                            
                            Zoom((0.85, 1.15)),       
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
        transform_test = transforms.Compose([
                            transforms.CenterCrop(hw_size),                       
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
    else:
        hw_size = 28
        transform_train = transforms.Compose([
                            transforms.ToTensor(),       
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
        transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])
    
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}       
    
    
                
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_target_dir, train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_target_dir, train=False, transform=transform_test),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader    


def load_data(data_aug, batch_size,workers,dataset, data_target_dir):
    
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        
    elif dataset == 'svhn':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    else:
        assert False, "Unknow dataset : {}".format(dataset)
    
    if data_aug==1:
        if dataset == 'svhn':
            train_transform = transforms.Compose(
                                             [ transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                            [transforms.ToTensor(), transforms.Normalize(mean, std)])
        else:
            train_transform = transforms.Compose(
                                                 [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        train_transform = transforms.Compose(
                                             [ transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
                                            [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif dataset == 'svhn':
        train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'stl10':
        train_data = datasets.STL10(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.STL10(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                         num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    
    return train_loader, test_loader, num_classes