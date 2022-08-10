import os
import torch
import utils

class Sample_Manager:

    def __init__(self, model, batch_count, freq, path=None, W=None, strategy='avg'):

        self.model = model
        self.idx = 0
        self.batch = 0
        self.batch_count = batch_count
        self.freq = freq
        self.milestones = [int(batch_count*(i+1)/freq) for i in range(freq)]

        self.W = W
        self.path = path
        self.disk = path is not None
        self.sample = self.sample_param_disk if self.disk else self.sample_param_mem
        
        self.min_loss = float('inf')
        self.max_loss = 0
        self.max_progress = 0

        self.param_vec = utils.get_model_param_vec(self.model)
        self.model_state = self.model.state_dict()
        self.mark_sample = self.mark_sample_disk if self.disk else self.mark_sample_mem

        if strategy == 'avg': self.strategy = self.strategy_avg_param
        elif strategy == 'max': self.strategy = self.strategy_max_loss
        elif strategy == 'min': self.strategy = self.strategy_min_loss
        elif strategy == 'pro': self.strategy = self.strategy_max_progress
        elif strategy == 'uni': self.strategy = self.strategy_uniform
        else: raise Exception('invalid sampling strategy')

    def get_samples(self):
        return self.W

    def get_last_samples(self, epochs):
        return self.W[:,self.idx-self.freq*epochs+1:]

    def mark_sample_disk(self):
        self.model_state = self.model.state_dict()

    def mark_sample_mem(self):
        self.param_vec = utils.get_model_param_vec(self.model)

    def step(self, criterion, inputs, labels, prev_loss):

        self.strategy(criterion, inputs, labels, prev_loss)
        self.batch = (self.batch % self.batch_count) + 1

        if self.batch in self.milestones:

            self.idx = self.idx + 1

            if self.strategy == self.strategy_avg_param:
                
                i = self.milestones.index(self.batch)
                n = self.batch - self.milestones[i-1] if i > 0 else self.batch
                self.param_vec = self.param_vec/n

                if self.disk:
                    state = self.model.state_dict()
                    utils.update_param(self.model, self.param_vec)
                    self.model_state = self.model.state_dict()
                    self.sample()
                    self.model.load_state_dict(state)
                else:
                    self.sample()

            else:
                self.sample()

            self.reset_values()

    def sample_param_disk(self):
            torch.save(self.model_state, os.path.join(self.path + '/checkpoint_' + str(self.idx)))

    def sample_param_mem(self):
        sample = torch.unsqueeze(self.param_vec, 1)
        self.W = torch.cat((self.W, sample), -1)

    def strategy_avg_param(self, criterion, inputs, labels, prev_loss):
        self.param_vec = self.param_vec + utils.get_model_param_vec(self.model)

    def strategy_max_loss(self, criterion, inputs, labels, prev_loss):

        outputs = self.model.forward(inputs)
        loss = criterion(outputs, labels)

        if loss > self.max_loss:
            self.max_loss = loss
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

    def reset_values(self):
        self.min_loss = float('inf')
        self.max_loss = 0
        self.max_progress = 0
        self.param_vec = utils.get_model_param_vec(self.model)
        self.model_state = self.model.state_dict()

    