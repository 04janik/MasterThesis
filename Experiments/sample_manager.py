import os
import torch

class Sample_Manager:

    def __init__(self, model, batch_count, freq, path=None, W=None, strategy='uni'):

        self.model = model
        self.idx = 0
        self.batch = 0
        self.batch_count = batch_count
        self.freq = freq
        self.strategy = strategy
        self.milestones = [int(batch_count*(i+1)/freq) for i in range(freq)]

        self.W = W
        self.path = path
        self.sample = self.sample_param_mem if path is None else self.sample_param_disk
        
        self.avg_loss = float('inf')
        self.min_loss = float('inf')
        self.max_loss = 0
        self.max_progress = 0

        self.param_vec = None
        self.model_state = None
        self.mark_sample = self.mark_sample_mem if path is None else self.mark_sample_disk

        match strategy:
            case 'avg': self.strategy = self.strategy_avg_loss()
            case 'max': self.strategy = self.strategy_max_loss()
            case 'min': self.strategy = self.strategy_min_loss()
            case 'pro': self.strategy = self.strategy_max_progress()
            case 'uni': self.strategy = self.strategy_uniform()
            case _: raise Exception('invalid sampling strategy')

    def get_samples(self):
        return self.W

    def get_last_samples(self, epochs):
        return self.W[:,self.idx-self.freq*epochs+1:]

    def get_model_param_vec(self):
        vec = []
        for name, param in self.model.named_parameters():
            vec.append(param.detach().reshape(-1))
        return torch.cat(vec, 0)

    def mark_sample_disk(self):
        self.model_state = self.model.state_dict()

    def mark_sample_mem(self):
        self.param_vec = self.get_model_param_vec()

    def step(self, criterion, inputs, labels, prev_loss):

        self.strategy(criterion, inputs, labels, prev_loss)
        self.batch = (self.batch % self.batch_count) + 1

        if self.batch in self.milestones:
            self.idx = self.idx + 1
            self.sample()
            self.reset_values()

    def sample_param_disk(self):
        torch.save(self.model_state, os.path.join(self.path + '/checkpoint_' + str(self.idx)))

    def sample_param_mem(self):
        sample = torch.unsqueeze(self.param_vec, 1)
        self.W = torch.cat((self.W, sample), -1)

    def strategy_avg_loss(self, criterion, inputs, labels, prev_loss):
        self.mark_sample()

    def strategy_max_loss(self, criterion, inputs, labels, prev_loss):
        if prev_loss > self.max_loss:
            self.max_loss = prev_loss
            self.mark_sample()

    def strategy_min_loss(self, criterion, inputs, labels, prev_loss):

        outputs = self.model.forward(inputs)
        loss = criterion(outputs, labels)

        if loss < self.min_loss:
            self.min_loss = loss
            self.mark_sample()

    def strategy_max_progress(self, criterion, inputs, labels, prev_loss):

        outputs = self.model.forward(inputs)
        loss = criterion(outputs, labels)
        progress = prev_loss - loss

        if progress > self.max_progress:
            self.max_progress = progress
            self.mark_sample()

    def strategy_uniform(self, criterion, inputs, labels, prev_loss):
        self.mark_sample()

    def reset_values():
        self.avg_loss = float('inf')
        self.min_loss = float('inf')
        self.max_loss = 0
        self.max_progress = 0
        self.param_vec = None
        self.model_state = None

    