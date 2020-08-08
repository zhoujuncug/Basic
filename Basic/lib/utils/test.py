import logging
import torch
import pprint
from collections import defaultdict
from lib.basic_tools.pause import *

kw = defaultdict(list)
def test(args, model, device, test_dataset, epoch, **kwargs):
    kw.update(kwargs)

    logger = logging.getLogger(__name__)
    model.eval()
    with torch.no_grad():
        test_loss0 = 0
        test_loss1 = 0
        for i, (data, target) in enumerate(test_dataset):
            data, target = data.to(device), target.to(device)
            y0, y1 = model(data)
            test_loss0 += torch.abs(y0 - target).sum()
            test_loss1 += torch.abs(y1 - target).sum()
    test_loss0 /= len(test_dataset.dataset)
    test_loss1 /= len(test_dataset.dataset)
    logger.info(pprint.pformat(
        f'Test: Average loss: {test_loss0:6.2f}, {test_loss1:6.2f}'
    ))

    # visualization

    if kw['viz'] is not None:
        kw['x'].append(epoch)
        kw['y'].append([test_loss0.item(), test_loss1.item()])

        opts1 = {
            "title": 'Test',
            "xlabel": 'epoch',
            "ylabel": 'depth_loss',
            "width": 300,
            "height": 200,
            "legend": ['y0', 'y1']
        }

        kw['viz'].line(X=kw['x'],
                 Y=kw['y'],
                 win='depth_loss',
                 opts=opts1)

    # pause
    pause(data, target, y0, y1)