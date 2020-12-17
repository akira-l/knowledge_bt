import os

import torch
import torch.nn as nn

from transformer.Models import Transformer
from model.Layers import EncoderLayer, DecoderLayer

import pdb


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1,):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, vis_feats, return_attns=False):

        enc_slf_attn_list = []
        enc_output = self.layer_norm(vis_feats)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,




class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, vis_feat, kg_feat, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.layer_norm(vis_feat)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, kg_feat)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,




class TransModel(nn.Module):
    def __init__(self, layer_num, head_num, dk_num,
                   dv_num, model_num, inner_num, dropout=0.1):
        super(TransModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_num, eps=1e-6)
        self.encoder = Encoder(n_layers = layer_num,
                               n_head = head_num,
                               d_k = dk_num,
                               d_v = dv_num,
                               d_model = model_num,
                               d_inner = inner_num,
                               )
        self.decoder = Decoder(n_layers = layer_num,
                              n_head = head_num,
                              d_k = dk_num,
                              d_v = dv_num,
                              d_model = model_num,
                              d_inner = inner_num,
                              )
        self.kg_trans_layer = nn.Linear(300, 2048)
        self.ans_trans = nn.Linear(2048, 128)
        self.ans_out = nn.Linear(128*10, 10)
        self.relu = nn.ReLU()

    def forward(self, vis_inputs, kg_inputs):
        bs = vis_inputs.size(0)
        encode_feat, *_ = self.encoder(vis_inputs)
        kg_trans = self.kg_trans_layer(kg_inputs)
        out, *_ = self.decoder(kg_trans, encode_feat)
        ans_trans_feat = self.ans_trans(out)
        ans_trans_feat = self.relu(ans_trans_feat)
        ans_out = self.ans_out(ans_trans_feat.view(bs, -1))
        return ans_out


