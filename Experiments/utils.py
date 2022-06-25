
import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

def get_datasets(args):

    if args.data == 'CIFAR10':

        # load CIFAR10 dataset
        cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        cifar10_mean = np.mean(cifar10_train.data/256, axis=(0,1,2))
        cifar10_std = np.std(cifar10_train.data/256, axis=(0,1,2))

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # configure dataloaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False)

        return train_loader, test_loader

    if args.data == 'CIFAR100':

        # load CIFAR100 dataset
        cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
        cifar100_mean = np.mean(cifar100_train.data/256, axis=(0,1,2))
        cifar100_std = np.std(cifar100_train.data/256, axis=(0,1,2))

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ])

        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

        # configure dataloaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False)

        return train_loader, test_loader

    raise Exception('invalid data set')


def get_model(args):

    if args.model == 'densenet121':
        from models.densenet import DenseNet121
        return DenseNet121()
    elif args.model == 'resnet8':
        from models.resnet import ResNet8
        return ReNet8()
    elif args.model == 'vgg11':
        from models.vgg import VGG11_bn
        return VGG11_bn()
    
    raise exception('invalid model')