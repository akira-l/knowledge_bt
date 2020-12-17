import os
import torch


if os.path.exists('../pretrained'):
    pretrained_model = {'resnet50' : '../pretrained/resnet50-19c8e357.pth',
                        'resnet101': './models/pretrained/se_resnet101-7e38fcc6.pth',
                        'senet154':'./models/pretrained/checkpoint_epoch_017_prec3_93.918_pth.tar'}
else:
    pretrained_model = {'resnet50' : './pretrained/resnet50-19c8e357.pth',
                        'resnet101': './models/pretrained/se_resnet101-7e38fcc6.pth',
                        'senet154':'./models/pretrained/checkpoint_epoch_017_prec3_93.918_pth.tar'}




class LoadConfig(object):
    def __init__(self, args, version):
        if version == 'train':
            get_list = ['train', 'val']
        elif version == 'val':
            get_list = ['val']
        elif version == 'test':
            get_list = ['test']
        elif version == 'ensamble':
            get_list = []
        else:
            raise Exception("train/val/test ???\n")

        self.save_dir = './net_model'
        self.train_bs = args.train_batch
        self.epoch = args.epoch
        self.val_bs = args.val_batch
        self.save_point = args.save_point
        self.check_point = args.check_point
        self.base_lr = args.base_lr
        self.lr_step = args.decay_step
        self.discription = args.discribe

        self.log_folder = './logs'
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)


