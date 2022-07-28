
import os
import torch 

# matrix calculations
from sklearn.decomposition import PCA
import numpy as np

# utility functions
from utils import get_datasets, get_model

# monitoring tools
from fastprogress import master_bar, progress_bar
import wandb
import time


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


def get_subspace(args, model):

    W = []

    # load sampled model parameters
    for i in range(args.sample_start, args.sample_start + args.samples):
        model.load_state_dict(torch.load(os.path.join(args.spath, 'checkpoint_' + str(i))))
        W.append(get_model_param_vec(model))

    W = torch.tensor(np.array(W)).cuda()
    print('W:', W.shape)

    # obtain subspace via PCA
    start = time.time()
    U, S, V = torch.pca_lowrank(W, q=args.dim, center=True)

    Q = torch.transpose(V,0,1).cuda()
    print('Q:', Q.shape)

    end = time.time()
    pca_time = end - start 
    print("PCA time consumed:", pca_time)

    return Q, pca_time


def train_PSGD(args, model, train_loader, test_loader):

    model.train()

    # configure training
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

    # get basis of subspace
    Q, pca_time = get_subspace(args, model)

    # start from specified model state
    model.load_state_dict(torch.load(os.path.join(args.spath, 'checkpoint_' + str(args.sample_start))))

    # construct name
    model_name = model.__class__.__name__
    run_name = f'{model_name}-PSGD-lr{args.lr}-d{args.dim}-s{args.samples}'

    # storage
    acc_max = 0 # max accuracy

    # configure monitoring tool
    with wandb.init(project=model_name, name=run_name) as run:

        # log some info
        run.log({'PCA time consumption': pca_time})
        run.config.learning_rate = args.lr
        run.config.optimizer = optimizer
        run.watch(model)

        # progress bar
        mb = master_bar(range(args.epochs))

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
            run.log({'accuracy': accuracy, 'max accuracy': acc_max, 'epoch': epoch, 'epoch time consumption': end - start})

            # schedule learning rate
            if args.lr_scheduler:
                lr_scheduler.step()


def train_BSGD(args, model, train_loader, test_loader):

    model.train()

    # configure training
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)

    # construct name
    model_name = model.__class__.__name__
    run_name = f'{model_name}-BSGD-lr{args.lr}'

    # storage
    acc_max = 0, d = 0, k = 0, Q = [], W = []

    # get sampling milestones
    batch_count = len(train_loader)
    milestones = [int(batch_count*(i+1)/args.xi) for i in range(args.xi)]

    # configure monitoring tool
    with wandb.init(project=model_name, name=run_name) as run:

        # log some info
        run.config.learning_rate = args.lr
        run.config.optimizer = optimizer
        run.watch(model)

        while k < args.epochs:

            # progress bar
            mb = master_bar(range(8))

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

            d = d+5

            W = torch.tensor(np.array(W)).cuda()
            print('W:', W.shape)

            # obtain subspace via PCA
            start = time.time()
            U, S, V = torch.pca_lowrank(W, q=d, center=True)
            end = time.time()
            pca_time = end - start 
            print("PCA time consumed:", pca_time)
            run.log({'PCA time consumption': pca_time})

            var_d = torch.sum(S)
            var_k = torch.sum(torch.diag(torch.mm(W.transpose(0,1), W)))

            print('variance of first ', d, ' components: ',var_d)
            print('variance of first ', k, ' components: ',var_k)

            if var_d > 0.99*var_k:
                Q = torch.transpose(V,0,1).cuda()
                print('Q:', Q.shape)
                break
        
        # progress bar
        mb = master_bar(range(args.epochs - k))

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
            run.log({'accuracy': accuracy, 'max accuracy': acc_max, 'epoch': epoch, 'epoch time consumption': end - start})

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

def train_SGD(args, model, train_loader, test_loader):

    model.train()

    # configure training
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)

    # schedule learning rate
    if args.data == 'CIFAR10':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
    elif args.data == 'CIFAR100':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.1)

    # construct name
    model_name = model.__class__.__name__
    run_name = f'{model_name}-SGD-lr{args.lr}'

    # construct path for checkpoints
    run_path = os.path.join(args.rpath, run_name + '-f' + str(args.freq))

    # save model state
    os.makedirs(run_path)
    torch.save(model.state_dict(), os.path.join(run_path + '/checkpoint_' + str(0)))

    # storage
    acc_max = 0 # max accuracy
    samples = 1 # number of sampled parameters

    # get sampling milestones
    batch_count = len(train_loader)
    milestones = [int(batch_count*(i+1)/args.freq) for i in range(args.freq)]

    # configure monitoring tool
    with wandb.init(project=model_name, name=run_name) as run:

        # log some info
        run.config.learning_rate = args.lr
        run.config.optimizer = optimizer
        run.watch(model)

        # progress bar
        mb = master_bar(range(args.epochs))

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
                    torch.save(model.state_dict(), os.path.join(run_path + '/checkpoint_' + str(samples)))
                    samples = samples + 1

            end = time.time()

            # evaluate the model
            accuracy, confusion = eval_model(args, model, test_loader)
            acc_max = acc_max if acc_max > accuracy else accuracy
            run.log({'accuracy': accuracy, 'max accuracy': acc_max, 'epoch': epoch, 'epoch time consumption': end - start})

            # schedule learning rate
            if args.lr_scheduler:
                lr_scheduler.step()


def eval_model(args, model, test_loader):

    train_mode = model.training

    if train_mode:
        model.eval()

    # set up confusion matrix
    dim = 10 if args.data == 'CIFAR10' else 100
    confusion = np.zeros((dim,dim), dtype=np.int32)

    # iterate test set
    for inputs, labels in iter(test_loader):

        inputs = inputs.cuda()
        outputs = model(inputs)

        for label, output in zip(labels, outputs.cpu().detach().numpy()):
            confusion[label, np.argmax(output)] += 1

    # compute accuracy
    total = np.sum(confusion)
    accuracy = np.sum(np.diag(confusion)) / total

    if train_mode:
        model.train()

    return accuracy, confusion