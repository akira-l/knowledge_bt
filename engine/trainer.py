import os,time,datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable

from model.transformer_model import TransModel
from data_loader.feat_loader import Featset

import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

class Trainer():
    def __init__(self, config, model, train_loader, val_loader):
        super(Trainer, self).__init__()

        self.epoch = config.epoch
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train_batch_size = train_loader.batch_size
        self.train_epoch_step = train_loader.__len__()

        self.check_point = config.check_point if config.check_point < self.train_epoch_step else self.train_epoch_step
        self.save_point = config.save_point if config.save_point < self.train_epoch_step else self.train_epoch_step

        self.get_ce_loss = nn.CrossEntropyLoss()
        self.get_nonreduce_celoss = nn.CrossEntropyLoss(reduction='none')
        self.get_kld_loss = nn.KLDivLoss(reduction='batchmean')
        self.get_l1_loss = nn.L1Loss()
        self.get_smooth_l1_loss = nn.SmoothL1Loss()
        self.get_sim_loss_noreduce = nn.CosineEmbeddingLoss(reduction='none')
        self.get_sim_loss = nn.CosineEmbeddingLoss()
        self.get_sim = nn.CosineSimilarity()

        self.model = model
        self.optim = optim.SGD(self.model.parameters(), lr=config.base_lr, momentum=0.9)
        self.scheduler = lr_scheduler.MultiStepLR(self.optim, milestones=[20, 40, 60, 80], gamma=0.1)

        self.epoch_num = 0
        self.step = 0

    def run(self, ):
        for epoch_iter in range(self.epoch):

            self.epoch_num = epoch_iter
            self.scheduler.step(epoch_iter)
            self.model.train(True)

            for batch_cnt, data in enumerate(self.train_loader):
                self.step += 1
                loss = 0
                self.model.train(True)

                vis_feats, kg_feats, labels = data
                vis_feats = Variable(vis_feats.cuda())
                kg_feats = Variable(kg_feats.cuda())
                labels = Variable(labels.cuda())

                #self.train(vis_feats, kg_feats, labels)
                ans_out = self.model(vis_feats, kg_feats)
                ce_loss = self.get_ce_loss(ans_out, labels)
                ce_loss.backward()

                self.optim.step()
                print('step: {:-8d} / {:d} loss: {:6.4f}'.format(
                            self.step , self.train_epoch_step, ce_loss.detach().item(),
                        ))


                if self.step % self.check_point == 0:
                    self.eval()


    def train(self, vis_feats, kg_feats, labels):
        ans_out = self.model(vis_feats, kg_feats)
        ce_loss = self.get_ce_loss(ans_out, labels)

        ce_loss.backward()

        self.optim.step()
        print('step: {:-8d} / {:d} loss: {:6.4f}'.format(
                    self.step , self.train_epoch_step, ce_loss.detach().item(),
                ))


    def eval(self, ):
        self.model.train(False)

        val_corrects1 = 0
        val_corrects2 = 0
        val_corrects3 = 0

        val_size = self.val_loader.__len__()
        item_count = self.val_loader.total_item_len
        t0 = time.time()

        val_batch_size = self.val_loader.batch_size
        val_epoch_step = self.val_loader.__len__()

        print('=='*10, '\n', 'evaluating ...')
        with torch.no_grad():
            for batch_cnt_val, data_val in enumerate(self.val_loader):
                vis_feats, kg_feats, labels = data_val
                vis_feats = Variable(vis_feats.cuda())
                kg_feats = Variable(kg_feats.cuda())
                labels = Variable(labels.cuda())

                # forward
                outputs = self.model(vis_feats, kg_feats)
                ce_loss = self.get_ce_loss(outputs, labels).detach().item()

                print('{:s} eval_batch: {:-6d} / {:d} loss: {:8.4f}'.format('val', batch_cnt_val, val_epoch_step, ce_loss))

                top3_val, top3_pos = torch.topk(outputs, 3)
                batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
                val_corrects1 += batch_corrects1
                batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
                val_corrects2 += (batch_corrects2 + batch_corrects1)
                batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
                val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)

            val_acc1 = val_corrects1 / item_count
            val_acc2 = val_corrects2 / item_count
            val_acc3 = val_corrects3 / item_count

            t1 = time.time()
            since = t1-t0
            print('--'*30)
            print('noraml eval: % 3d %s %s %s-loss: %.4f ||%s-acc@1: %.4f %s-acc@2: %.4f %s-acc@3: %.4f ||time: %d' % (self.epoch_num,
                                                                                                                       'val', dt(),
                                                                                                                       'val', ce_loss,
                                                                                                                       'val', val_acc1,
                                                                                                                       'val', val_acc2,
                                                                                                                       'val', val_acc3, since))
            print('--' * 30)








