import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


def get_model_param_vec(model):
    vec = []
    for name, param in model.named_parameters():
        vec.append(param.detach().reshape(-1))
    return torch.cat(vec, 0)


def get_model_grad_vec(model):
    vec = []
    for name, param in model.named_parameters():
        vec.append(param.grad.detach().reshape(-1))
    return torch.cat(vec, 0)


def update_grad(model, grad_vec):
    idx = 0
    for name, param in model.named_parameters():
        shape = param.grad.shape
        size = 1
        for i in range(len(list(shape))):
            size *= shape[i]
        param.grad.data = grad_vec[idx:idx+size].reshape(shape)
        idx += size


def update_param(model, param_vec):
    idx = 0
    for name, param in model.named_parameters():
        shape = param.data.shape
        size = 1
        for i in range(len(list(shape))):
            size *= shape[i]
        param.data = param_vec[idx:idx+size].reshape(shape)
        idx += size


def get_datasets(args):

    if args.data == 'CIFAR10':

        # load CIFAR10 dataset
        cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        cifar10_mean = np.mean(cifar10_train.data/255, axis=(0,1,2))
        cifar10_std = np.std(cifar10_train.data/255, axis=(0,1,2))

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        # configure dataloaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False)

        return train_loader, test_loader

    if args.data == 'CIFAR100':

        # load CIFAR100 dataset
        cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
        cifar100_mean = np.mean(cifar100_train.data/255, axis=(0,1,2))
        cifar100_std = np.std(cifar100_train.data/255, axis=(0,1,2))

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ])

        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

        # configure dataloaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False)

        return train_loader, test_loader

    raise Exception('invalid data set')


def get_model(args):

    if args.model == 'densenet121':
        from models.densenet import make_DenseNet121
        return make_DenseNet121()
    elif args.model == 'efficientnet':
        from models.efficientnet import make_EfficientNet
        return make_EfficientNet()
    elif args.model == 'googlenet':
        from models.googlenet import make_GoogLeNet
        return make_GoogLeNet()
    elif args.model == 'inceptionv3':
        from models.inceptionv3 import make_Inceptionv3
        return make_Inceptionv3()
    elif args.model == 'mobilenet':
        from models.mobilenet import make_MobileNet
        return make_MobileNet()
    elif args.model == 'nasnet':
        from models.nasnet import make_NasNet
        return make_NasNet()
    elif args.model == 'resnet8':
        from models.resnet import make_ResNet8
        return make_ResNet8()
    elif args.model == 'seresnet18':
        from models.seresnet import make_SEResNet18
        return make_SEResNet18()
    elif args.model == 'shufflenetv2':
        from models.shufflenetv2 import make_ShuffleNetv2
        return make_ShuffleNetv2()
    elif args.model == 'squeezenet':
        from models.squeezenet import make_SqueezeNet
        return make_SqueezeNet()
    elif args.model == 'vgg11':
        from models.vgg import make_VGG11_bn
        return make_VGG11_bn()
    elif args.model == 'xception':
        from models.xception import make_Xception
        return make_Xception()
    
    raise Exception('invalid model')