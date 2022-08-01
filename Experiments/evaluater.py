import numpy as np

class Evaluater:

    def __init__(self, model, test_loader, data_set):

        self.model = model
        self.test_loader = test_loader

        self.acc = 0
        self.acc_max = 0
        self.confusion = []
        self.epoch = 0
        self.targets = 10 if data_set == 'CIFAR10' else 100

    def eval_model(self, run):

        self.epoch = self.epoch + 1

        train_mode = self.model.training

        if train_mode:
            self.model.eval()

        self.confusion = np.zeros((self.targets,self.targets), dtype=np.int32)

        for inputs, labels in iter(self.test_loader):

            inputs = inputs.cuda()
            outputs = self.model(inputs)

            for label, output in zip(labels, outputs.cpu().detach().numpy()):
                self.confusion[label, np.argmax(output)] += 1

        self.acc = np.sum(np.diag(self.confusion))/np.sum(self.confusion)
        self.acc_max = self.acc if self.acc > self.acc_max else self.acc_max

        if train_mode:
            self.model.train()

        run.log({'accuracy': self.acc, 'max accuracy': self.acc_max, 'epoch': self.epoch})