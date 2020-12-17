import os
import json
import numpy as np
import random

import torch
import torch.nn as nn
import torch.utils.data as Data

import pdb

class Featset(Data.Dataset):
    def __init__(self, sample='full'):

        vis_feat_name_list = ['train2014', 'val2014']
        feat_root_path = '/data03/liangyzh/bottomup_feature'
        full_data_list = []
        except_list = ['COCO_val2014_000000096923.jpg', 'COCO_train2014_000000028645.jpg']
        for folder in vis_feat_name_list:
            folder_path = os.path.join(feat_root_path, folder)
            name_list = os.listdir(folder_path)
            for name in name_list:
                if name.strip('.npz') in except_list:
                    continue
                feat_path = os.path.join(folder_path, name)
                full_data_list.append(feat_path)

        print('full list len: ', len(full_data_list))

        if isinstance(sample, str):
            self.iid_to_frcn_feat_path = full_data_list

        if isinstance(sample, int):
            self.iid_to_frcn_feat_path = full_data_list[:sample]


        kg_vec_path = '/data03/liangyzh/numberbatch/vgdet_vector.pt'
        self.kg_vec_dict = torch.load(kg_vec_path)
        kg_vec_sim_path = '/data03/liangyzh/numberbatch/sim_dict.pt'
        self.kg_vec_sim = torch.load(kg_vec_sim_path)

        self.gen_iter = 0


    def proc_img_feat(self, img_feat, img_feat_pad_size):

        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return torch.from_numpy(img_feat).float()

    def proc_bbox_feat(self, bbox, img_shape):
        bbox_feat = np.zeros((bbox.shape[0], 5), dtype=np.float32)

        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[0])
        bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])

        return bbox_feat


    def load_img_feats(self, iid):
        frcn_feat = np.load(self.iid_to_frcn_feat_path[iid])
        frcn_feat_x = frcn_feat['x'].transpose((1, 0))
        bbox = self.proc_bbox_feat(
                frcn_feat['bbox'],
                (frcn_feat['image_h'], frcn_feat['image_w'])
            )

        if 'e/train' in self.iid_to_frcn_feat_path[iid]:
            file_name = self.iid_to_frcn_feat_path[iid].replace('ture/train', 'ture/vgdet_train').replace('.npz', '.npy')
        else:
            file_name = self.iid_to_frcn_feat_path[iid].replace('ture/val', 'ture/vgdet_val').replace('.npz', '.npy')

        cls_name = list(np.load(file_name))
        return frcn_feat_x, bbox, cls_name, len(cls_name)


    def get_sim_list(self, cur_cls, ind_len=10):
        sim_score = self.kg_vec_sim[cur_cls]['scores']
        sim_names = self.kg_vec_sim[cur_cls]['names']
        sort_ind = sorted(range(len(sim_score)), key=lambda k: sim_score[k], reverse=True)
        get_sort_ind = sort_ind[:ind_len]
        get_sort_name = [sim_names[x] for x in get_sort_ind]
        return get_sort_name


    def gen_ans_cls(self, cls_list):
        ans = []
        for sub_cls in cls_list:
            sim_list = self.get_sim_list(sub_cls)
            sample_cls = random.sample(sim_list, 1)
            ans += sample_cls
        return ans


    def gen_ins_sample(self, ins_ind, bbox, cls_name, ans_len=10):
        gt_ans = [cls_name[x] for x in ins_ind]
        gt_ans_ind = random.randint(0, ans_len-1)
        ans_list = []
        addition_iter = 0
        while ans_len != 1:
            fake_ans = self.gen_ans_cls(gt_ans)
            if fake_ans != gt_ans:
                ans_len = ans_len - 1
                ans_list.append(fake_ans)

        ans_list.insert(gt_ans_ind, gt_ans)
        return ans_list, gt_ans_ind, ins_ind


    def gen_motif_sample(self, ins_ind, bbox, cls_name, ins_size, ans_len=10):
        if 4 > ins_size and 2 < ins_size:
            motif_num = random.randint(2, ins_size)
            ans_list, ans_ind, gt_ind = self.gen_single_motif_sample(motif_num, ins_ind, bbox, cls_name)
        elif 2 > ins_size:
            ans_list, ans_ind, gt_ind = self.gen_ins_sample(ins_ind, bbox, cls_name)
        else:
            motif_num = random.randint(2, 4)
            ans_list, ans_ind, gt_ind = self.gen_single_motif_sample(motif_num, ins_ind, bbox, cls_name)
        return ans_list, ans_ind, gt_ind


    def gen_scene_sample(self, scene_num, cls_name, ins_size, ans_len=10):
        ans_list = []
        gt_ans_ind = random.randint(0, ans_len - 1)
        gt_inds = random.sample(list(range(len(cls_name))), scene_num)
        gt_ans = [cls_name[x] for x in gt_inds]

        while ans_len != 1:
            fake_ans = self.gen_ans_cls(gt_ans)
            if fake_ans != gt_ans:
                ans_len = ans_len - 1
                ans_list.append(fake_ans)
        ans_list.insert(gt_ans_ind, gt_ans)
        return ans_list, gt_ans_ind, gt_inds


    def gen_single_motif_sample(self, motifs_num, ins_ind, bbox, cls_name, ans_len=10):
        cen_x = (bbox[:, 0] + bbox[:, 2]) / 2
        cen_y = (bbox[:, 1] + bbox[:, 3]) / 2
        dis = (cen_x - cen_x[ins_ind])**2 + (cen_y - cen_y[ins_ind])**2
        sort_ind = dis.argsort()[1:motifs_num]
        cur_inds = dis.argsort()[:motifs_num].tolist()
        cur_clses = [cls_name[x] for x in sort_ind]
        gt_ans = [cls_name[ins_ind]] + cur_clses
        gt_ans_ind = random.randint(0, ans_len - 1)
        ans_list = []
        addition_iter = 0
        while ans_len != 1:
            fake_ans = self.gen_ans_cls(gt_ans)
            if fake_ans != gt_ans:
                ans_len = ans_len - 1
                ans_list.append(fake_ans)
        ans_list.insert(gt_ans_ind, gt_ans)
        return ans_list, gt_ans_ind, cur_inds


    def gen_sample(self, iid):
        feat, bbox, cls_name, ins_size = self.load_img_feats(iid)

        gen_iter = random.sample(['instance', 'motif', 'scene'], 1)[0]
        if gen_iter == 'instance':
            ins_sample_id = random.randint(0, ins_size -1)
            anses, ans_gt_ind, vis_ind = self.gen_ins_sample([ins_sample_id], bbox, cls_name)

        elif gen_iter == 'motif':
            ins_sample_id = random.randint(0, ins_size -1)
            anses, ans_gt_ind, vis_ind = self.gen_motif_sample(ins_sample_id, bbox, cls_name, ins_size)

        elif gen_iter == 'scene':
            if ins_size > 5:
                scene_num = random.randint(5, 10)
                anses, ans_gt_ind, vis_ind = self.gen_scene_sample(scene_num, cls_name, ins_size)
            else:
                ins_sample_id = random.randint(0, ins_size -1)
                anses, ans_gt_ind, vis_ind = self.gen_motif_sample(ins_sample_id, bbox, cls_name, ins_size)

        return anses, ans_gt_ind, vis_ind,\
               feat, bbox, cls_name, ins_size

    def __getitem__(self, item):
        ans_set, gt_set, vis_ind, feat, bbox, cls_name, ins_size = self.gen_sample(item)
        vis_feat = feat[vis_ind, :]
        vis_pad_feat = self.proc_img_feat(vis_feat, 10)

        ans_feat_set = []
        for sub_ans in ans_set:
            sub_ans_len = len(sub_ans)
            sub_ans_feat = torch.zeros(300)
            for ans in sub_ans:
                sub_ans_feat += self.kg_vec_dict[ans]
            sub_ans_feat = sub_ans_feat / sub_ans_len
            ans_feat_set.append(sub_ans_feat)

        ans_feat = torch.stack(ans_feat_set)

        return vis_pad_feat, ans_feat, gt_set


    def __len__(self):
        return len(self.iid_to_frcn_feat_path)



def collate_func(batch):
    label = []
    vis_feat_gather = []
    kg_feat_gather = []
    for sample in batch:
        vis_feat_gather.append(sample[0])
        kg_feat_gather.append(sample[1])
        label.append(sample[2])
    return torch.stack(vis_feat_gather, 0),\
           torch.stack(kg_feat_gather, 0),\
           torch.LongTensor(label)





if __name__ == '__main__':
    feat_set = Featset()

    for cnt in range(10):
        feat_set.test__getitem__(random.randint(0, 1000))




