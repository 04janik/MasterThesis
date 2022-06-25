"""
Implementation of VGG11/13/16/19.

Reference:
[1] Karen Simonyan, Andrew Zisserman:
    Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv 1409.1556.
"""


import torch
import torch.nn as nn
import numpy as np


cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, features, num_class=100):

        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output


def make_layers(cfg, batch_norm=False):

    layers = []
    channels_in = 3

    for layer in cfg:

        if layer == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(channels_in, layer, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(layer)]

        layers += [nn.ReLU(inplace=True)]
        channels_in = layer

    return nn.Sequential(*layers)


def VGG11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))


def VGG13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))


def VGG16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))


def VGG19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))