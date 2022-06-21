'''
Properly implemented ResNets for CIFAR10 as described in section 4.2 of [1]:

|------------|---------|---------|
| model      | #layers | #params |
|------------|---------|---------|
| ResNet20   |    20   |  0.27M  |
| ResNet32   |    32   |  0.46M  |
| ResNet44   |    44   |  0.66M  |
| ResNet56   |    56   |  0.85M  |
| ResNet110  |   110   |  1.70M  |
| ResNet1202 |  1202   | 19.40M  |
|------------|---------|---------|

This implementation is based on the work in [3].

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: 
    Deep Residual Learning for Image Recognition. arXiv:1512.03385.
[2] Frithjof Gressmann, Zach Eaton-Rosen, Carlo Luschi:
    Improving Neural Netwok Training in Low Dimensional Random Bases. arXiv:2011.04720.
[3] Tao Li, Lei Tan, Qinghua Tao, Yipeng Liu, Xiaolin Huang:
    https://github.com/nblt/DLDR/blob/main/resnet.py
'''


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BuildingBlock(nn.Module):

    def __init__(self, channels_in, channels_out, stride=1):

        super().__init__()

        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels_out)

        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels_out)

        if stride != 1 or channels_in != channels_out:
            self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, channels_out//4, channels_out//4), "constant", 0))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):

        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = F.relu(residual)

        residual = self.conv2(residual)
        residual = self.bn2(residual)

        output = residual + self.shortcut(x)
        output = F.relu(output)

        return output


class ResNet(nn.Module):

    def __init__(self, stack_size, num_classes=10):

        super().__init__()

        self.channels_in = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.stack1 = self.make_stack(channels_out=16, size=stack_size[0], stride=1)
        self.stack2 = self.make_stack(channels_out=32, size=stack_size[1], stride=2)
        self.stack3 = self.make_stack(channels_out=64, size=stack_size[2], stride=2)

        self.pool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, num_classes)

        self.apply(init_params)

    def make_stack(self, channels_out, size, stride):

        blocks = []
        strides = [stride] + [1]*(size-1)

        for stride in strides:
            blocks.append(BuildingBlock(self.channels_in, channels_out, stride))
            self.channels_in = channels_out

        return nn.Sequential(*blocks)

    def forward(self, x):

        output = self.conv1(x)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.stack1(output)
        output = self.stack2(output)
        output = self.stack3(output)

        output = self.pool(output)
        output = self.linear(output.view(output.size(0), -1))

        return output


def init_params(module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight)


def ResNet8(num_classes=10):
    return ResNet([1, 1, 1], num_classes)
 

def ResNet20(num_classes=10):
    return ResNet([3, 3, 3], num_classes)


def ResNet32(num_classes=10):
    return ResNet([5, 5, 5], num_classes)


def ResNet44(num_classes=10):
    return ResNet([7, 7, 7], num_classes)


def ResNet56(num_classes=10):
    return ResNet([9, 9, 9], num_classes)


def ResNet110(num_classes=10):
    return ResNet([18, 18, 18], num_classes)


def ResNet1202(num_classes=10):
    return ResNet([200, 200, 200], num_classes)


def count_params(net):

    params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        params += np.prod(x.data.numpy().shape)

    print("#params:", params)
    print("#layers:", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))

    '''
    #params in ResNet8 according to [1]: 75290
    #params in ResNet8 according to [2]: 78330

    That's a difference of 3040 between the two papers.
    The difference is exactly the same for any ResNet.

    |--------|-----------------------|
    | layer  | #params               |
    |--------|-----------------------|
    | conv1  |   464 = (3x3x3+2)*16  |
    | sec1   |                       |
    |  conv1 |  2336 = (3x3x16+2)*16 |
    |  conv2 |  2336 = (3x3x16+2)*16 |
    | sec2   |                       |
    |  conv1 |  4672 = (3x3x16+2)*32 |
    |  conv2 |  9280 = (3x3x32+2)*32 |
    | sec3   |                       |
    |  conv1 | 18560 = (3x3x32+2)*64 |
    |  conv2 | 36992 = (3x3x64+2)*64 |
    | linear |   650 = (64+1)*10     |
    |--------|-----------------------|
    |    SUM | 75290                 |
    |--------|-----------------------|
    '''


if __name__ == "__main__":
    for net_name in ['ResNet8', 'ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202']:
        print(net_name)
        count_params(globals()[net_name]())
        print()