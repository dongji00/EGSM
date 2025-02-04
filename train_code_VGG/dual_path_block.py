import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class BN_Con2d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride,
                 padding=0, groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BN_Con2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))

class DPN_Block(nn.Module):
    """
    Dual Path block
    """

    def __init__(self, in_chnls, add_chnl, cat_chnl, cardinality, d, stride):
        super(DPN_Block, self).__init__()
        self.add = add_chnl
        self.cat = cat_chnl
        self.chnl = cardinality * d
        self.conv1 = BN_Con2d(in_chnls, self.chnl, 1, 1, 0)
        self.conv2 = BN_Con2d(self.chnl, self.chnl, 3, stride, 1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.chnl, add_chnl + cat_chnl, 1, 1, 0)
        self.bn = nn.BatchNorm2d(add_chnl + cat_chnl)
        self.shortcut = nn.Sequential()
        if add_chnl != in_chnls:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chnls, add_chnl, 1, stride, 0),
                nn.BatchNorm2d(add_chnl)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        add = out[:, :self.add, :, :] + self.shortcut(x)
        out = torch.cat((add, out[:, self.add:, :, :]), dim=1)
        return F.relu(out)


class DPN(nn.Module):
    def __init__(self, blocks: object, add_chnls: object, cat_chnls: object,
                 conv1_chnl, cardinality, d, num_classes) -> object:
        super(DPN, self).__init__()
        self.cdty = cardinality
        self.chnl = conv1_chnl
        self.conv1 = BN_Con2d(3, self.chnl, 7, 2, 3)
        d1 = d
        self.conv2 = self.__make_layers(blocks[0], add_chnls[0], cat_chnls[0], d1, 1)
        d2 = 2 * d1
        self.conv3 = self.__make_layers(blocks[1], add_chnls[1], cat_chnls[1], d2, 2)
        d3 = 2 * d2
        self.conv4 = self.__make_layers(blocks[2], add_chnls[2], cat_chnls[2], d3, 2)
        d4 = 2 * d3
        self.conv5 = self.__make_layers(blocks[3], add_chnls[3], cat_chnls[3], d4, 2)
        self.fc = nn.Linear(self.chnl, num_classes)

    def __make_layers(self, block, add_chnl, cat_chnl, d, stride):
        layers = []
        strides = [stride] + [1] * (block-1)
        for i, s in enumerate(strides):
            layers.append(DPN_Block(self.chnl, add_chnl, cat_chnl, self.cdty, d, s))
            self.chnl = add_chnl + cat_chnl
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 3, 2, 1)
        print("shape 1---->", out.shape)
        out = self.conv2(out)
        print("shape 2---->", out.shape)
        out = self.conv3(out)
        print("shape 3---->", out.shape)
        out = self.conv4(out)
        print("shape 4---->", out.shape)
        out = self.conv5(out)
        print("shape 5---->", out.shape)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.softmax(out)