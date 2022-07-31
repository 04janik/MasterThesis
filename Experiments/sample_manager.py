import os
import torch

class Sample_Manager:

    def __init__(self, model, batch_count, freq, path=None, W=None):

        self.model = model
        self.idx = 0
        self.batch = 0
        self.batch_count = batch_count
        self.milestones = [int(batch_count*(i+1)/freq) for i in range(freq)]

        self.W = W
        self.path = path
        self.sample = self.sample_param_mem if path is None else self.sample_param_disk  

    def sample(self):
        self.sample()

    def step(self):

        if self.batch in self.milestones:
            self.sample()
            self.idx = self.idx + 1

        self.batch = self.batch + 1 % self.batch_count

    def sample_param_disk(self):
        torch.save(model.state_dict(), os.path.join(path + '/checkpoint_' + str(self.idx)))

    def sample_param_mem(self):
        sample = torch.unsqueeze(self.get_model_param_vec(), 1)
        self.W = torch.cat((self.W, sample), -1)

    def get_samples(self):
        return self.W

    def get_last_samples(self, epochs):
        return self.W[:,self.W.shape[1]-self.freq*epochs]

    def get_model_param_vec(self):
        vec = []
        for name, param in self.model.named_parameters():
            vec.append(param.detach().reshape(-1))
        return torch.cat(vec, 0)