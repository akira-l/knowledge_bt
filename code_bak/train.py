
import os
import math
import time
from tqdm import tqdm
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

import pdb

def train():

    transformer = Transformer(
        n_src_vocab = 1315, #opt.src_vocab_size,
        n_trg_vocab = 10, #opt.trg_vocab_size,
        src_pad_idx = opt.src_pad_idx,
        trg_pad_idx = opt.trg_pad_idx,
        trg_emb_prj_weight_sharing = False,
        emb_src_trg_weight_sharing = False,
        d_k = 64,
        d_v = 64,
        d_model = 512,
        d_word_vec = 512,
        d_inner = 2048,
        n_layers = 6,
        n_head = 8,
        dropout = 0.1)




if __name__ == '__main__':
    train()
