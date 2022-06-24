
# neural network stuff
import torch 
import torch.nn as nn
import torch.optim as optim

# dataset loading
import torchvision
import torchvision.transforms as transforms

# matrix calculations
from sklearn.decomposition import PCA
import numpy as np

# the objective model
from models.resnet import ResNet8

# monitoring tools
from fastprogress import master_bar, progress_bar
import wandb

# helpers
import os
import random
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-bs', default=128, type=int, help='batch size')                    # batch size
parser.add_argument('-dim', default=15, type=int, help='subspace dimension')            # subspace dimension
parser.add_argument('-epochs', default=80, type=int, help='number of epochs')           # epochs
parser.add_argument('-lr', default=0.1, type=float, help='learning rate')               # learning rate
parser.add_argument('-mom', default=0.9, type=float, help='momentum for sgd')           # momentum
parser.add_argument('-optimizer', default='sgd', type=str, help='choose sgd/psgd')      # optimizer
parser.add_argument('-rs', default=22, type=int, help='random seed')                    # random seed
parser.add_argument('-samples', default=30, type=int, help='number of sampling epochs') # sampling epochs
parser.add_argument('-rpath', default='', type=str, help='result path')                 # result path
parser.add_argument('-spath', default='', type=str, help='sampling path')               # sampling path
parser.add_argument('-wd', default=1e-4, type=float, help='weight decay')               # weight decay

args = parser.parse_args()
random.seed(args.rs)

# check arguments
if args.bs <= 0:
    raise Exception('invalid batch size')
elif args.dim <= 0:
    raise Exception('invalid subspace dimension')
elif args.epochs <= 0:
    raise Exception('invalid epochs')
elif args.lr <= 0:
    raise exception('invalid learning rate')
elif args.mom < 0 or args.mom > 1:
    raise Exception('invalid momentum')
elif not (args.optimizer == 'sgd' or args.optimizer == 'psgd'):
    raise Exception('invalid optimizer')
elif args.samples < args.dim:
    raise exception('invalid sampling epochs')
elif not os.path.exists(args.rpath):
    raise Exception('invalid result path')
elif args.optimizer == 'psgd' and not os.path.exists(args.spath):
    raise Exception('invalid sampling path')
elif args.wd < 0:
    raise exception('invalid weight decay')

# login to monitoring tool
wandb.login(key='147686d07ab47cb770a0957694c8a6f896671f2c')

# configure device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device}')

# enable cudnn.benchmark
if device == 'cuda':
    cudnn.benchmark = True

# configure model
model = ResNet8().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)

# load CIFAR10 dataset
cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
cifar10_mean = np.mean(cifar10_train.data/256, axis=(0,1,2))
cifar10_std = np.std(cifar10_train.data/256, axis=(0,1,2))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# configure dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False)


def get_model_param_vec(model):

    vec = []

    for name, param in model.named_parameters():
        vec.append(param.cpu().detach().numpy().reshape(-1))

    return np.concatenate(vec, 0)


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


def train_PSGD():

    model.train()

    # load sampled model parameters
    W = []
    for i in range(args.samples):
        model.load_state_dict(torch.load(os.path.join(args.spath, 'checkpoint_' + str(i))))
        W.append(get_model_param_vec(model))
    w = np.array(W)

    # obtain base variables via PCA
    pca = PCA(args.dim)
    pca.fit_transform(W)

    P = np.array(pca.components_)
    P = torch.from_numpy(P).to(device)

    # start from initial model state
    model.load_state_dict(torch.load(os.path.join(args.spath, 'checkpoint_' + str(0))))

    # construct name
    model_name = model.__class__.__name__
    optimizer_name = 'P' + optimizer.__class__.__name__
    run_name = f'{model_name}-{optimizer_name}-lr{args.lr}-d{args.dim}'

    # check if same experiment was run before
    run_path = os.path.join(args.rpath, run_name + '-' + str(0))
    run_index = 0

    while os.path.exists(run_path):
        run_index += 1
        run_path = os.path.join(args.rpath, run_name + '-' + str(run_index))

    # save model state
    os.makedirs(run_path)
    torch.save(model.state_dict(), os.path.join(run_path + '/checkpoint_' + str(0)))

    # configure monitoring tool
    with wandb.init(project="ResNet8", name=run_name) as run:

        # log some info
        run.config.learning_rate = args.lr
        run.config.optimizer = optimizer
        run.watch(model)

        # progress bar
        mb = master_bar(range(args.epochs))

        for epoch in mb:

            for inputs, labels in progress_bar(iter(train_loader), parent=mb):

                # move data to device
                inputs, labels = inputs.to(device), labels.to(device)

                # forward pass
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)

                # backward pass in subspace
                optimizer.zero_grad()
                loss.backward()

                grad = get_model_grad_vec(model).double()
                grad = torch.mm(P, grad.reshape(-1,1))
                grad_pro = torch.mm(P.transpose(0,1), grad)

                update_grad(model, grad_pro)
                optimizer.step()

                # log the loss
                run.log({'loss:': loss})

            # evaluate the model after every epoch
            accuracy, per_class_accuracy, confusion = test_model()
            mb.main_bar.comment = f'val acc:{accuracy}'
            run.log({'accuracy': accuracy, 'epoch': epoch})

            # save model state after every epoch
            torch.save(model.state_dict(), os.path.join(run_path + '/checkpoint_' + str(epoch+1)))


def train_SGD():

    model.train()

    # construct name
    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    run_name = f'{model_name}-{optimizer_name}-lr{args.lr}'

    # check if same experiment was run before
    run_path = os.path.join(args.rpath, run_name + '-' + str(0))
    run_index = 0

    while os.path.exists(run_path):
        run_index += 1
        run_path = os.path.join(args.rpath, run_name + '-' + str(run_index))

    # save model state
    os.makedirs(run_path)
    torch.save(model.state_dict(), os.path.join(run_path + '/checkpoint_' + str(0)))

    # configure monitoring tool
    with wandb.init(project="ResNet8", name=run_name) as run:

        # log some info
        run.config.learning_rate = args.lr
        run.config.optimizer = optimizer
        run.watch(model)

        # progress bar
        mb = master_bar(range(args.epochs))

        for epoch in mb:

            for inputs, labels in progress_bar(iter(train_loader), parent=mb):

                # move data to device
                inputs, labels = inputs.to(device), labels.to(device)

                # forward pass
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log the loss
                run.log({'loss:': loss})

            # evaluate the model after every epoch
            accuracy, per_class_accuracy, confusion = test_model()
            mb.main_bar.comment = f'val acc:{accuracy}'
            run.log({'accuracy': accuracy, 'epoch': epoch})

            # save model state after every epoch
            torch.save(model.state_dict(), os.path.join(run_path + '/checkpoint_' + str(epoch+1)))


def test_model():

    train_mode = model.training

    if train_mode:
        model.eval()

    # set up confusion matrix
    confusion = np.zeros((10,10), dtype=np.int32)

    # iterate test set
    for inputs, labels in iter(test_loader):

        inputs = inputs.to(device)
        outputs = model(inputs)

        for label, output in zip(labels, outputs.cpu().detach().numpy()):
            confusion[label, np.argmax(output)] += 1

    # compute some metrics
    total = np.sum(confusion)
    accuracy = np.sum(np.diag(confusion)) / total
    per_class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)

    if train_mode:
        model.train()

    return accuracy, per_class_accuracy, confusion


def print_test_results():
    accuracy, per_class_accuracy, confusion = test_model()
    print(f'Global accuracy{accuracy:.2%}')
    print('Confusion matrix:')
    print(confusion)
    print('Per class accuracies:')
    for acc, name in zip(per_class_accuracy, train_set.classes):
        print(f'{name:>10}: {acc:.2%}')


if args.optimizer == 'sgd':
    train_SGD()
elif args.optimizer == 'psgd':
    train_PSGD()

print_test_results()


