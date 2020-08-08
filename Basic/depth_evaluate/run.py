import argparse
import pprint
import torch.optim as optim
import visdom

from lib.basic_tools.log import *
from lib.basic_tools.device import *
from lib.basic_tools.reproducibility import *
from lib.dataset.data4depth import *
from lib.models.depth_eval import *
from lib.utils.train import *
from lib.utils.test import *

parser = argparse.ArgumentParser(description='Basic Training')
# train
parser.add_argument('--data_augment', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--milestones', type=list, default=[5, 10, 15, 20, 25])
parser.add_argument('--gamma', type=float, default=0.1)
# visualization
parser.add_argument('--print_freq', type=int, default=1000)
parser.add_argument('--visdom', type=bool, default=True)
# reproducibility
parser.add_argument('--seed', type=list, default=(1, 1))
parser.add_argument('--save_model', type=bool, default=True)

args = parser.parse_args()

# logging
logger, output_dir = create_logger()
logger.info(pprint.pformat(args))

# random seed
if args.seed[0]:
    set_seed(args.seed[1])

device = is_cuda()

# dataloader
train_path = '../data/data4depth_train150.npy'
test_path = '../data/data4depth_eval.npy'
train_dataset = DataLoader(
                            MyDataset(train_path, is_train=True, is_augment=True),
                            batch_size=args.batch_size,
                            num_workers=1,
                            pin_memory=True,
                            shuffle=True,
                            )
test_dataset = DataLoader(
                            MyDataset(test_path, is_train=True, is_augment=True),
                            batch_size=args.batch_size,
                            num_workers=1,
                            pin_memory=True,
                            shuffle=False,
                            )

# model
model = DepthEvaluator().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
# scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# viz
kwargs = {'viz': None}
if args.visdom:
    viz = visdom.Visdom(env='depth eval')
    kwargs.update({
        'viz': viz,
    })

# training
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_dataset, optimizer, epoch, **kwargs)

    test(args, model, device, test_dataset, epoch, **kwargs)

    scheduler.step()