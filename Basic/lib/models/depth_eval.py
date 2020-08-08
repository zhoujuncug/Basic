import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 0)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0)
        self.conv3 = nn.Conv2d(64, 128, 11, 1, 0)
        self.conv4 = nn.Conv2d(128, 1, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = x.mean(2).mean(2)
        return x.reshape(-1)


class DepthEvaluator(nn.Module):
    def __init__(self):
        super(DepthEvaluator, self).__init__()
        self.net1 = Net()
        self.net2 = Net()

    def forward(self, x):
        x0 = (x - 3000) * (1. / 3000)
        y0 = self.net1(x0) * 3000 + 3000
        r = 250
        x1 = torch.clamp(x - y0[:, None, None, None], -r, r) * (1. / r)
        y1 = self.net2(x1) * r + y0

        return y0, y1