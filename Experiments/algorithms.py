import os
import torch
import numpy as np
import math

import utils
from evaluater import Evaluater
from sample_manager import Sample_Manager

from fastprogress import master_bar, progress_bar
import wandb
import time


def pca(W):

    start = time.time()

    # centralize the samples
    means = torch.mean(W, dim=0)
    W = W - means.expand_as(W)

    # perform spectral decomposition
    WTW = torch.mm(W.transpose(0,1), W)
    S, V = torch.linalg.eigh(WTW)
    S = torch.sqrt(S)

    end = time.time()
    pca_time = end - start
    print("PCA time consumed:", pca_time)

    return S, V, pca_time


def pc_angles(V1, V2):

    if V1.shape != V2.shape:
        raise Exception('Angle computation failed: invalid shapes provided')

    angles = []

    for i in range(V1.shape[1]):

        thetas = []

        for j in range(V2.shape[1]):
            theta = torch.acos(torch.dot(V1[:,i],V2[:,j]))
            if abs(theta > math.pi/2):
                theta = theta + math.pi/2 if theta < 0 else theta - math.pi/2
            thetas.append(theta.item())

        angles.append(np.min(thetas))

    return angles


def get_best_dim(S, precision=0.95):

    d = 0
    d_max = S.shape[0]

    pca_var_d = 0
    pca_var = torch.sum(S)

    while pca_var_d < precision*pca_var:
        d = d+1
        pca_var_d = pca_var_d + S[d_max-d]

    print('#samples:', d_max)
    print('dimension:', d)

    return d


def get_subspace(args, model):

    # load the initialization
    model.load_state_dict(torch.load(os.path.join(args.spath, 'checkpoint_' + str(0))))
    W = torch.unsqueeze(utils.get_model_param_vec(model), 1)

    # load sampled parameters
    for i in range(args.samples):
        model.load_state_dict(torch.load(os.path.join(args.spath, 'checkpoint_' + str(i+1))))
        sample = torch.unsqueeze(utils.get_model_param_vec(model), 1)
        W = torch.cat((W, sample), -1)

    W = W.cuda()

    # perform PCA
    S, V, pca_time = pca(W)

    # determine basis of subspace
    idx = args.samples - args.dim + 1
    Q = torch.mm(W,V[:,idx:])
    Q = torch.div(Q,S[idx:])
    Q = Q.cuda()

    print('W:', W.shape)
    print('Q:', Q.shape)

    return Q, pca_time


def train_SGD_epoch(model, criterion, optimizer, train_loader, run, master_bar, sample_manager=None, evaluater=None):

    start = time.time()

    for inputs, labels in progress_bar(iter(train_loader), parent=master_bar):

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

        # sample parameters
        if sample_manager is not None:
            sample_manager.step(criterion, inputs, labels, loss)

    run.log({'epoch time consumption': time.time() - start})


def train_SGD(args, model, train_loader, test_loader):

    model.train()

    # construct name
    model_name = model.__class__.__name__
    run_name = f'{model_name}-SGD-lr{args.lr}'

    # create directory for checkpoints
    run_path = os.path.join(args.rpath, run_name + '-' + args.strat + '-f' + str(args.freq))
    os.makedirs(run_path)

    # define sample manager and sample initialization
    sample_manager = Sample_Manager(model, len(train_loader), freq=args.freq, path=run_path, strategy=args.strat)
    sample_manager.sample()

    # configure training
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    evaluater = Evaluater(model, criterion, test_loader, args.data)

    # schedule learning rate
    if args.data == 'CIFAR10':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
    elif args.data == 'CIFAR100':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.1)

    # configure monitoring tool
    with wandb.init(project=model_name, name=run_name) as run:

        run.watch(model)

        # progress bar
        mbar = master_bar(range(args.epochs))

        for epoch in mbar:

            # train for one epoch
            train_SGD_epoch(model, criterion, optimizer, train_loader, run, mbar, sample_manager, evaluater)

            # evaluate the model
            evaluater.eval_model(run)

            # schedule learning rate
            if args.lr_scheduler:
                lr_scheduler.step()


def train_PSGD_epoch(Q, model, criterion, optimizer, train_loader, run, master_bar):

    start = time.time()

    for inputs, labels in progress_bar(iter(train_loader), parent=master_bar):

        # move data to cuda
        inputs, labels = inputs.cuda(), labels.cuda()

        # forward pass
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # parameter update in subspace
        grad = utils.get_model_grad_vec(model).float()
        grad = torch.mm(Q.transpose(0,1), grad.reshape(-1,1))
        grad = torch.mm(Q, grad)
        utils.update_grad(model, grad)
        optimizer.step()

        # log the loss
        run.log({'loss': loss})

    run.log({'epoch time consumption': time.time() - start})


def train_PSGD(args, model, train_loader, test_loader):

    model.train()

    # construct name
    model_name = model.__class__.__name__
    run_name = f'{model_name}-PSGD-lr{args.lr}-d{args.dim}-s{args.samples}'

    # get basis of subspace
    Q, pca_time = get_subspace(args, model)

    # load the initialization
    model.load_state_dict(torch.load(os.path.join(args.spath, 'checkpoint_' + str(0))))

    # project initialization to subspace
    param = utils.get_model_param_vec(model).float()
    param = torch.mm(Q.transpose(0,1), param.reshape(-1,1))
    param = torch.mm(Q, param)
    utils.update_param(model, param)

    # configure training
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
    evaluater = Evaluater(model, criterion, test_loader, args.data)

    # configure monitoring tool
    with wandb.init(project=model_name, name=run_name) as run:

        run.watch(model)
        run.log({'PCA time consumption': pca_time})

        # progress bar
        mbar = master_bar(range(args.epochs))

        for epoch in mbar:

            # train for one epoch
            train_PSGD_epoch(Q, model, criterion, optimizer, train_loader, run, mbar)

            # evaluate the model
            evaluater.eval_model(run)

            # schedule learning rate
            if args.lr_scheduler:
                lr_scheduler.step()


def train_BSGD(args, model, train_loader, test_loader):

    model.train()

    # construct name
    model_name = model.__class__.__name__
    run_name = f'{model_name}-BSGD-lr{args.lr}'

    # configure training
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    evaluater = Evaluater(model, criterion, test_loader, args.data)

    # define sample manager
    sample_manager = Sample_Manager(model, len(train_loader), freq=args.freq, W=torch.unsqueeze(utils.get_model_param_vec(model), 1), strategy=args.strat)

    # variables
    S_old = None
    V_old = None
    Q = None
    k = 0

    # configure monitoring tool
    with wandb.init(project=model_name, name=run_name) as run:

        run.watch(model)

        while k < args.epochs:

            # progress bar
            mbar = master_bar(range(5))

            for epoch in mbar:

                # train for one epoch
                train_SGD_epoch(model, criterion, optimizer, train_loader, run, mbar, sample_manager)

                # evaluate the model
                evaluater.eval_model(run)

                k = k+1

            # get samples from last 5 epochs
            W = sample_manager.get_last_samples(epochs=5).cuda()

            # perform PCA
            S, V, pca_time = pca(W)

            run.log({'sigma_1': S[4]})
            run.log({'sigma_2': S[3]})
            run.log({'sigma_3': S[2]})
            run.log({'sigma_4': S[1]})
            run.log({'sigma_5': S[0]})

            if V_old is not None:

                # compute angles between current
                # and previous principal components
                angles = pc_angles(V_old, V)

                if np.max(angles) < math.pi/180 and torch.max(torch.div(S,S_old)) < 1.02:

                    # select all samples
                    W = sample_manager.get_samples().cuda()

                    # perform PCA
                    S, V, pca_time = pca(W)

                    # get dimension
                    d = get_best_dim(S, 0.95)
                    idx = k - d + 1

                    # determine basis
                    Q = torch.mm(W,V[:,idx:])
                    Q = torch.div(Q,S[idx:])
                    Q = Q.cuda()

                    print('W:', W.shape)
                    print('Q:', Q.shape)
                    break

            S_old = S
            V_old = V

        # project parameters to subspace
        param = utils.get_model_param_vec(model).float()
        param = torch.mm(Q.transpose(0,1), param.reshape(-1,1))
        param = torch.mm(Q, param)
        utils.update_param(model, param)

        for lr in [args.lr, 0.1*args.lr]:

            print('Training 10 epochs of PSGD with learning rate:', lr)

            # define optimizer for PSGD
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.mom)

            # progress bar
            mbar = master_bar(range(10))

            for epoch in mbar:

                # train for one epoch
                train_PSGD_epoch(Q, model, criterion, optimizer, train_loader, run, mbar)

                # evaluate the model
                evaluater.eval_model(run)

                k = k+1


        '''
        # progress bar
        mbar = master_bar(range(args.epochs - k))

        stuck = 0
        acc_psgd = 0

        for epoch in mbar:

            # train for one epoch
            train_PSGD_epoch(Q, model, criterion, optimizer, train_loader, run, mbar)

            # evaluate the model
            evaluater.eval_model(run)

            k = k+1

            if evaluater.acc > acc_psgd:
                acc_psgd = evaluater.acc
            else:
                stuck = stuck + 1

            if stuck == 3:
                print('PSGD converged -> switching to SGD')
                break

        # define optimizer for SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1*args.lr, momentum=args.mom)

        # progress bar
        mbar = master_bar(range(args.epochs - k))

        for epoch in mbar:

            # train for one epoch
            train_SGD_epoch(model, criterion, optimizer, train_loader, run, mbar)

            # evaluate the model
            evaluater.eval_model(run)
            '''


'''
def get_subspace(args, model):

    W = []

    # load sampled model parameters
    for i in range(args.sample_start, args.sample_start + args.samples):
        model.load_state_dict(torch.load(os.path.join(args.spath, 'checkpoint_' + str(i))))
        W.append(get_model_param_vec(model))

    W = np.array(W)
    print('W:', W.shape)

    # obtain subspace via PCA
    start = time.time()
    pca = PCA(args.dim)
    pca.fit_transform(W)

    Q = np.array(pca.components_)
    Q = torch.from_numpy(Q).cuda()
    print('Q:', Q.shape)

    end = time.time()
    pca_time = end - start
    print("PCA time consumed:", pca_time)

    return Q, pca_time
'''


'''
def get_subspace(args, model):

    # load the initialization
    model.load_state_dict(torch.load(os.path.join(args.spath, 'checkpoint_' + str(0))))
    W = torch.unsqueeze(get_model_param_vec(model), 1)

    # load sampled parameters
    for i in range(args.samples):
        model.load_state_dict(torch.load(os.path.join(args.spath, 'checkpoint_' + str(i+1))))
        sample = torch.unsqueeze(get_model_param_vec(model), 1)
        W = torch.cat((W, sample), -1)

    W = W.cuda()
    print('W:', W.shape)

    start = time.time()
    U, S, V = torch.pca_lowrank(W, q=args.dim, center=True)

    Q = U.cuda()
    print('Q:', Q.shape)

    end = time.time()
    pca_time = end - start 
    print("PCA time consumed:", pca_time)

    return Q, pca_time
'''


'''
def train_BSGD(args, model, train_loader, test_loader):

    model.train()

    # configure training
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)

    # construct name
    model_name = model.__class__.__name__
    run_name = f'{model_name}-BSGD-lr{args.lr}-d{args.dim}-s{args.samples}'

    # storage
    k = 0 # number of epochs
    W = [] # sampled parameters
    acc_max = 0 # max accuracy

    # get sampling milestones
    batch_count = len(train_loader)
    milestones = [int(batch_count*(i+1)/args.xi) for i in range(args.xi)]

    # configure monitoring tool
    with wandb.init(project=model_name, name=run_name) as run:

        run.config.learning_rate = args.lr
        run.config.optimizer = optimizer
        run.watch(model)

        for i in range(args.rho):

            print('subspace refinement: ', i+1)

            # progress bar
            mb = master_bar(range(int(2*args.dim/(args.xi*(args.rho)))))

            for epoch in mb:

                start = time.time()

                batch = 0

                for inputs, labels in progress_bar(iter(train_loader), parent=mb):

                    # count batches
                    batch = batch + 1

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

                    # sample parameters
                    if batch in milestones:
                        W.append(get_model_param_vec(model))

                end = time.time()

                # evaluate the model
                accuracy, confusion = eval_model(args, model, test_loader)
                acc_max = acc_max if acc_max > accuracy else accuracy
                run.log({'accuracy': accuracy, 'max accuracy': acc_max, 'epoch': epoch, 'epoch time consumption': end - start})

                k = k+1

            # obtain subspace via PCA
            W = torch.tensor(np.array(W)).cuda()
            dim = int(args.dim*(i+1)/(args.rho))

            print('W:', W.shape)
            print('dim:', dim)

            U, S, V = torch.pca_lowrank(W, q=dim, center=True)

            Q = torch.transpose(V,0,1).cuda()
            print('Q:', Q.shape)

            j = 0
            acc_prev = 0

            print('Training in low-dimensional subspace...')

            while j<3:

                start = time.time()

                for inputs, labels in iter(train_loader):

                    # move data to cuda
                    inputs, labels = inputs.cuda(), labels.cuda()

                    # forward pass
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)

                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()

                    # parameter update in subspace
                    grad = get_model_grad_vec(model).float()
                    grad = torch.mm(Q, grad.reshape(-1,1))
                    grad_pro = torch.mm(Q.transpose(0,1), grad)

                    update_grad(model, grad_pro)
                    optimizer.step()

                    # log the loss
                    run.log({'loss': loss})

                end = time.time()
                
                # evaluate the model
                accuracy, confusion = eval_model(args, model, test_loader)
                acc_max = acc_max if acc_max > accuracy else accuracy

                if accuracy > acc_prev:
                    acc_prev = accuracy
                else:
                    j = j+1

                run.log({'accuracy': accuracy, 'max accuracy': acc_max, 'epoch': epoch, 'epoch time consumption': end - start})

                k = k+1

        # progressbar 
        mb = master_bar(range(k,args.epochs))

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
            accuracy, confusion = eval_model(args, model, test_loader)
            acc_max = acc_max if acc_max > accuracy else accuracy
            run.log({'accuracy': accuracy, 'max accuracy': acc_max, 'epoch': epoch, 'epoch time consumption': end - start})
'''