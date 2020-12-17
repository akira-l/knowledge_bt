import os
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from model.transformer_model import TransModel
from data_loader.feat_loader import Featset, collate_func
from engine.trainer import Trainer
from config import LoadConfig

import argparse

import pdb


os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description='feature bert parameters')
    parser.add_argument('--save', dest='resume',
                        default=None, #,
                        type=str)
    parser.add_argument('--epoch', dest='epoch',
                        default=360, type=int)
    parser.add_argument('--tb', dest='train_batch',
                        default=512, type=int)
    parser.add_argument('--vb', dest='val_batch',
                        default=128, type=int)
    parser.add_argument('--sp', dest='save_point',
                        default=5000, type=int)
    parser.add_argument('--cp', dest='check_point',
                        default=5000, type=int)
    parser.add_argument('--lr', dest='base_lr',
                        default=0.001, type=float)
    parser.add_argument('--lr_step', dest='decay_step',
                        default=60, type=int)
    parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio',
                        default=10.0, type=float)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=0,  type=int)
    parser.add_argument('--tnw', dest='train_num_workers',
                        default=8, type=int)
    parser.add_argument('--vnw', dest='val_num_workers',
                        default=8, type=int)
    parser.add_argument('--detail', dest='discribe',
                        default='', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('\nargs:  ', args, '\n', '========'*6)
    Config = LoadConfig(args, 'train')
    print('Config:  ', vars(Config), '\n', '========'*6)

    model = TransModel(layer_num=2,
                       head_num=4,
                       dk_num=64,
                       dv_num=64,
                       model_num=2048,
                       inner_num=512,
                       )
    model.cuda()
    model = nn.DataParallel(model)
    train_set = Featset(sample='full')
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size = args.train_batch,
                                               shuffle=True,
                                               num_workers=args.train_num_workers,
                                               collate_fn=collate_func,
                                               drop_last=False,
                                               pin_memory=True)
    setattr(train_loader, 'total_item_len', len(train_set))
    val_set = Featset(sample=1000)
    val_loader = torch.utils.data.DataLoader(val_set,
                                               batch_size = args.val_batch,
                                               shuffle=False,
                                               num_workers=args.val_num_workers,
                                               collate_fn=collate_func,
                                               drop_last=False,
                                               pin_memory=True)
    setattr(val_loader, 'total_item_len', len(val_set))
    cudnn.benchmark = True

    time = datetime.datetime.now()
    filename = '%s_%d%d%d_'%(args.discribe, time.month, time.day, time.hour)
    save_dir = os.path.join(Config.save_dir, filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    trainer = Trainer(Config, model, train_loader, val_loader)
    trainer.run()

    pdb.set_trace()



