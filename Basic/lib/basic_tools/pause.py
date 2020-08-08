import os
import pprint
import matplotlib.pyplot as plt
import torch
import torchvision


def pause(data, target, y0, y1):
    if not os.path.isfile('pause.txt'):
        os.mknod('pause.txt')
    file = open('pause.txt')
    is_pause = True if file.read() else False
    if is_pause:
        img = torchvision.utils.make_grid(data, nrow=8, padding=2)
        img = img[0, ...].to('cpu')
        print('target', target)
        print('y0', y0.clone().detach())
        print('y1', y1.clone().detach())
        print('E0', torch.abs(y0-target).clone().detach())
        print('E1', torch.abs(y1-target).clone().detach())
        plt.imshow(img)
        plt.show()