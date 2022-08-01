import argparse
import numpy
import os
import random
import torch
import wandb

import algorithms
from utils import get_datasets, get_model

# define parser
parser = argparse.ArgumentParser()

# parse arguments
parser.add_argument('-model', default='', type=str, help='choose the model')                # model 
parser.add_argument('-data', default='', type=str, help='choose CIFAR10/CIFAR100')          # data set
parser.add_argument('-optimizer', default='', type=str, help='choose sgd/psgd/bsgd')        # optimizer
parser.add_argument('-epochs', default=0, type=int, help='number of epochs')                # epochs
parser.add_argument('-lr', default=0, type=float, help='learning rate')                     # learning rate
parser.add_argument('-lr_scheduler', default=True, type=bool, help='lr scheduler')          # lr scheduler
parser.add_argument('-seed', default=None, type=int, help='random seed')                    # random seed
parser.add_argument('-wd', default=1e-4, type=float, help='weight decay')                   # weight decay
 
# parse arguments for sgd
parser.add_argument('-freq', default=1, type=int, help='sampling frequency per epoch')      # sampling frequency
parser.add_argument('-rpath', default='', type=str, help='result path')                     # result path
parser.add_argument('-strat', default='uni', type=str, help='sampling strategy')            # sampling strategy

# parse arguments for psgd
parser.add_argument('-dim', default=0, type=int, help='subspace dimension')                 # subspace dimension
parser.add_argument('-samples', default=0, type=int, help='number of sampling epochs')      # sampling epochs
parser.add_argument('-spath', default='', type=str, help='sampling path')                   # sampling path

# store arguments
args = parser.parse_args()
args.bs = 128
args.mom = 0.9

# check arguments
if args.epochs <= 0:
    raise Exception('invalid epochs')
elif args.lr <= 0:
    raise Exception('invalid learning rate')
elif args.wd < 0:
    raise Exception('invalid weight decay')
elif args.bs <= 0:
    raise Exception('invalid batch size')
elif args.mom < 0 or args.mom > 1:
    raise Exception('invalid momentum')

if args.optimizer == 'sgd':
    if args.freq <=0:
        raise Exception('invalid sampling frequency')
    elif not os.path.exists(args.rpath):
        raise Exception('invalid result path')
    elif args.strat is not in ['avg', 'max', 'min', 'pro', 'uni']:
        raise Exception('invalid sampling strategy')

elif args.optimizer == 'psgd':
    if args.dim <= 0 or args.dim > args.samples:
        raise Exception('invalid subspace dimension')
    elif args.samples <= 0:
        raise Exception('invalid number of samples')
    elif not os.path.exists(args.spath):
        raise Exception('invalid sampling path')

elif args.optimizer == 'bsgd':
    if args.freq <= 0:
        raise Exception('invalid sampling frequency')
    elif args.strat is not in ['avg', 'max', 'min', 'pro', 'uni']:
        raise Exception('invalid sampling strategy')

else:
    raise Exception('invalid optimizer')

# set random seed
if args.seed is not None:
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

# login to monitoring tool
wandb.login(key='147686d07ab47cb770a0957694c8a6f896671f2c')

# check cuda accesss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {device}')

# define training task
model = get_model(args).cuda()
train_loader, test_loader = get_datasets(args)

# train model
if args.optimizer == 'sgd':
    algorithms.train_SGD(args, model, train_loader, test_loader)
elif args.optimizer == 'psgd':
    algorithms.train_PSGD(args, model, train_loader, test_loader)
elif args.optimizer == 'bsgd':
    algorithms.train_BSGD(args, model, train_loader, test_loader)

print('Succesfully trained the model. Check results on wandb.ai.')