
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
from models.resnet import make_ResNet8

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

# set random seeds
args = parser.parse_args()
random.seed(args.rs)
np.random.seed(args.rs)
torch.manual_seed(args.rs)
torch.cuda.manual_seed(args.rs)

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

# login to monitoring tool
wandb.login(key='147686d07ab47cb770a0957694c8a6f896671f2c')

# check cuda access
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device}')

# configure model
model = make_ResNet8().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)

# load CIFAR10 dataset
cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
cifar10_mean = np.mean(cifar10_train.data/255, axis=(0,1,2))
cifar10_std = np.std(cifar10_train.data/255, axis=(0,1,2))

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

transfom_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# configure dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False)


def get_model_param_vec(model):
    vec = []
    for name, param in model.named_parameters():
        vec.append(param.detach().cpu().numpy().reshape(-1))
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
    for i in range(args.sample_start, args.sample_start + args.samples):
        model.load_state_dict(torch.load(os.path.join(args.spath, 'checkpoint_' + str(i))))
        W.append(get_model_param_vec(model))
    W = np.array(W)
    print('W:', W.shape)

    # obtain subspace via PCA
    start = time.time()
    pca = PCA(args.dim)
    pca.fit_transform(W)

    P = np.array(pca.components_)
    P = torch.from_numpy(P).cuda()

    end = time.time()
    pca_time = end - start
    print("PCA time consumed:", pca_time)

    # start from initial model state
    model.load_state_dict(torch.load(os.path.join(args.spath, 'checkpoint_' + str(0))))

    # construct name
    model_name = model.__class__.__name__
    optimizer_name = 'P' + optimizer.__class__.__name__
    run_name = f'{model_name}-{optimizer_name}-lr{args.lr}-d{args.dim}-s{args.samples}'

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
        run.log({'PCA time consumption': pca_time})
        run.config.learning_rate = args.lr
        run.config.optimizer = optimizer
        run.watch(model)

        # progress bar
        mb = master_bar(range(args.epochs))

        # store max accuracy
        acc_max = 0

        for epoch in mb:

            start = time.time()

            for inputs, labels in progress_bar(iter(train_loader), parent=mb):

                # move data to cuda
                inputs, labels = inputs.cuda() labels.cuda()

                # forward pass
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)

                # backward pass
                optimizer.zero_grad()
                loss.backward()

                # parameter update in subspace
                grad = get_model_grad_vec(model).float()
                grad = torch.mm(P, grad.reshape(-1,1))
                grad_pro = torch.mm(P.transpose(0,1), grad)

                update_grad(model, grad_pro)
                optimizer.step()

                # log the loss
                run.log({'loss': loss})

            end = time.time()

            # evaluate the model
            accuracy, per_class_accuracy, confusion = eval_model()
            acc_max = acc_max if acc_max > accuracy else accuracy
            run.log({'accuracy': accuracy, 'max accuracy': acc_max, 'epoch': epoch, 'epoch time consumption': end - start})

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

        # store max accuracy
        acc_max = 0

        for epoch in mb:

            start = time.time()

            for inputs, labels in progress_bar(iter(train_loader), parent=mb):

                # move data to cuda
                inputs, labels = inputs.cuda(), labels.cuda()

                # forward pass
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log the loss
                run.log({'loss': loss})

            end = time.time()

            # evaluate the model
            accuracy, per_class_accuracy, confusion = eval_model()
            acc_max = acc_max if acc_max > accuracy else accuracy
            run.log({'accuracy': accuracy, 'max accuracy': acc_max, 'epoch': epoch, 'epoch time consumption': end - start})

            # save model state after every epoch
            torch.save(model.state_dict(), os.path.join(run_path + '/checkpoint_' + str(epoch+1)))


def eval_model():

    train_mode = model.training

    if train_mode:
        model.eval()

    # set up confusion matrix
    confusion = np.zeros((10,10), dtype=np.int32)

    # iterate test set
    for inputs, labels in iter(test_loader):

        inputs = inputs.cuda()
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
    print(f'Global accuracy: {accuracy:.2%}')
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


