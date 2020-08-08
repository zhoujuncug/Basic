import logging
import pprint
from easydict import EasyDict as edict
from collections import defaultdict
import torch.nn.functional as F
from lib.basic_tools.pause import *

kw = defaultdict(list)
def train(args, model, device, train_loader, optimizer, epoch, **kwargs):
    kw.update(kwargs)
    logger = logging.getLogger(__name__)
    model.train()
    loss_all0 = 0
    loss_all1 = 0
    for batch_i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        y0, y1 = model(data)
        loss0 = F.l1_loss(y0, target)
        loss1 = F.l1_loss(y1, target)
        loss = loss0 + loss1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all0 += loss0
        loss_all1 += loss1

        # print
        if batch_i % args.print_freq == 0:
            logger.info(pprint.pformat(
                f'Train Epoch: {epoch:<3,d} '
                f'[{(batch_i+1) * len(data):<5,d} / {len(train_loader.dataset):<5d}'
                f'({(batch_i+1) / len(train_loader):.1%})]'
                f'    Loss: {loss.item():6.2f}'
            ))

        # pause
        pause(data, target, y0, y1)

    # visualization
    loss_all0 = loss_all0 / len(train_loader.dataset)
    loss_all1 = loss_all1 / len(train_loader.dataset)
    if kw['viz'] is not None and epoch > 1:
        kw['x'].append(epoch)
        kw['y'].append([loss_all0.item(), loss_all1.item()])

        opts1 = {
            "title": 'Train',
            "xlabel": 'epoch',
            "ylabel": 'training_loss',
            "width": 300,
            "height": 200,
            "legend": ['y0', 'y1']
        }

        kw['viz'].line(X=kw['x'],
                 Y=kw['y'],
                 win='train_loss')

