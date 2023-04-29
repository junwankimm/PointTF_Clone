import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.transforms import transforms as T
import argparse
from torch.utils.data import DataLoader
from util import *
from module import *
import os

##
parser = argparse.ArgumentParser(description='train PointTransformer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', type=float, default=1e-4, dest='lr')
parser.add_argument('--batch_size', type=int, default=8, dest='batch_size')
parser.add_argument('--num_epoch', type=int, default=50, dest='num_epoch')
parser.add_argument('--device', type=str, default='cpu', dest='device')
parser.add_argument('mode', type=str, default='train', dest='mode')

args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
mode = args.mode
device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'mps' if args.device == 'mps' and torch.backends.mps.is_available() else 'cpu'
##
data_dir = './datasets'
result_dir = './results'
ckpt_dir = './ckpt'
log_dir = './log'

print('mode {}'.format(mode))
##
if not os.path.exists(data_dir):
    raise ValueError('No Dataset Directory')

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'npy'))
##
if mode == 'train':
    pass
##

