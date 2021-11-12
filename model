import torch
import os
import math
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import spacy


class Positional_Emb(torch.nn.Module):
  def __init__(self,seq_len, emb_dim, dropout = 0):
    super(Positional_Emb,self).__init__()
    self.PE = np.array([[i / np.power(10000, 2*(j//2)/emb_dim) for j in range(emb_dim)] for i in range(seq_len)])
    self.PE[:, 0::2] = np.sin(self.PE[:, 0::2])
    self.PE[:, 1::2] = np.cos(self.PE[:, 1::2])
    self.PE = torch.from_numpy(self.PE)
    self.dropout = torch.nn.Dropout(p = dropout)

  def forward(self,x):
    pos_emb = self.PE.repeat(x.shape[0],1,1)
    out = x + pos_emb.type_as(x)
    out = self.dropout(out)
    return out


class Attention(torch.nn.Module):
  def __init__(self, emb_dim, att_dim, mask = False, dropout = 0):
    super(Attention, self).__init__()
    self.att_dim = att_dim
    self.mask = mask
    self.key_w = torch.nn.Linear(emb_dim, att_dim, bias = False)
    self.query_w = torch.nn.Linear(emb_dim, att_dim, bias = False)
    self.value_w = torch.nn.Linear(emb_dim, att_dim, bias = False)
    self.softmax = torch.nn.Softmax(dim = 2)
    self.dropout = torch.nn.Dropout(p = dropout)

  def forward(self, q, k, v, mask_pad):
    q, k, v = self.query_w(q), self.key_w(k), self.value_w(v)
    qk = torch.bmm(q, k.permute(0,2,1)) / math.sqrt(self.att_dim)
    qk = qk.masked_fill(mask_pad == 0, -1e20)
    if self.mask == True:
      qk = torch.tril(qk)
      qk = qk.masked_fill(qk == 0, -1e20)
    att_w = self.dropout(self.softmax(qk))
    out = torch.bmm(att_w, v)
    return out
    
class Multi_Head_ATT(torch.nn.Module):
    def __init__(self, emb_dim, multi_head = 1, mask = False, post_LN = True, dropout = 0):
      super(Multi_Head_ATT,self).__init__()
      self.head = multi_head
      self.post_LN = post_LN
      self.att = torch.nn.ModuleList([
                                      Attention(emb_dim, emb_dim//multi_head, mask = mask, dropout = dropout) for i in range(multi_head)
                                      ])
      self.LN = torch.nn.LayerNorm(emb_dim, eps = 1e-6)
      self.WO = torch.nn.Linear(emb_dim, emb_dim)#, bias = False)
      self.dropout = torch.nn.Dropout(p = dropout)

    def forward(self, q,k,v, mask_pad):
      res = q
      if self.post_LN == False:
        q, k, v = self.LN(q), self.LN(k), self.LN(v)
      if self.head == 1:
        out = self.att[0](q, k, v, mask_pad) 
      else:
        out = self.att[0](q, k, v, mask_pad)
        for i in range(1, self.head):
          outi = self.att[i](q, k, v, mask_pad)
          out = torch.cat([out, outi], dim = -1)
        out = self.WO(out)   
      out = self.dropout(out)
      if self.post_LN == False:
        out = out + res
      else:
        out = self.LN(out + res)
      return out

class Feed_Forward(torch.nn.Module): 
  def __init__(self, emb_dim, dim_expan = 4, post_LN = True, dropout = 0):
    super(Feed_Forward,self).__init__()
    self.w1 = torch.nn.Linear(emb_dim, dim_expan*emb_dim)
    self.w2 = torch.nn.Linear(dim_expan*emb_dim, emb_dim)
    self.relu = torch.nn.ReLU()
    self.LN = torch.nn.LayerNorm(emb_dim, eps = 1e-6)
    self.dropout = torch.nn.Dropout(p = dropout)
    self.post_LN = post_LN
  def forward(self,x):
    if self.post_LN == False:
      x = self.LN(x)
    out = self.relu(self.w1(x))
    out = self.w2(out)
    out = self.dropout(out)
    if self.post_LN == False:
      out = out + x
    else:
      out = self.LN(out + x)
    return out

class Encoder(torch.nn.Module):
  def __init__(self, num_layer, emb_dim, head, dim_expan = 4, post_LN = True, dropout = 0):
    super(Encoder, self).__init__()
    self.num_layer = num_layer
    self.attention = Multi_Head_ATT( emb_dim, multi_head = head, dropout = dropout)
    self.FF = Feed_Forward(emb_dim, dim_expan = dim_expan, dropout = dropout)
    self.connect1 = torch.nn.ModuleList([
                                         Multi_Head_ATT(emb_dim, multi_head = head, post_LN = post_LN, dropout = dropout) for i in range(num_layer - 1)
                                         ])
    self.connect2 = torch.nn.ModuleList([
                                         Feed_Forward(emb_dim, dim_expan = dim_expan, post_LN = post_LN, dropout = dropout) for i in range(num_layer - 1)
                                         ])

  def forward(self, x, mask_pad):
    out = self.FF(self.attention(x, x, x, mask_pad))
    for idx in range(self.num_layer - 1):
      out = self.connect1[idx](out, out, out, mask_pad)
      out = self.connect2[idx](out)
    return out

class Decoder(torch.nn.Module):
  def __init__(self,num_layer, emb_dim, head, dim_expan = 4, post_LN = True, dropout = 0):
    super(Decoder,self).__init__()
    self.num_layer = num_layer
    self.attention = Multi_Head_ATT(emb_dim, multi_head= head, mask = True, dropout = dropout)
    self.cross_att = Multi_Head_ATT(emb_dim, multi_head= head, dropout = dropout)
    self.FF = Feed_Forward(emb_dim, dim_expan = dim_expan, dropout = dropout)
    self.connect_1 = torch.nn.ModuleList([
                                          Multi_Head_ATT(emb_dim, multi_head= head, mask = True, post_LN = post_LN, dropout = dropout) for i in range(self.num_layer - 1)
                                          ])
    self.connect_2 = torch.nn.ModuleList([
                                          Multi_Head_ATT(emb_dim, multi_head= head, post_LN = post_LN, dropout = dropout) for i in range(self.num_layer -1)
                                          ])
    self.connect_3 = torch.nn.ModuleList([
                                          Feed_Forward(emb_dim, dim_expan = dim_expan, post_LN = post_LN, dropout = dropout) for i in range(self.num_layer -1)
                                          ]) 

  def forward(self, x, enc_out, mask_pad, c_att_pad):
    out = self.attention(x, x, x, mask_pad) # in q, k ,v
    out = self.FF(self.cross_att(out, enc_out, enc_out, c_att_pad))
    if self.num_layer > 1:
      for idx in range(self.num_layer - 1):
        out = self.connect_1[idx](out, out, out, mask_pad)
        out = self.connect_2[idx](out, enc_out, enc_out, c_att_pad) # q from decoder input, k,v from encoder output
        out = self.connect_3[idx](out)
    return out

class Transformer(torch.nn.Module):
  def __init__(self,num_layer , seq_len, emb_dim, head, word_size_in, word_size_out, dim_expan = 4, post_LN = True,dropout = 0):
    super(Transformer,self).__init__() 
    self.num_layer = num_layer
    self.emb_dim = emb_dim
    self.pos_emb = Positional_Emb(seq_len, emb_dim, dropout= dropout)
    self.embedding_in = torch.nn.Embedding(word_size_in, emb_dim, padding_idx= 1)
    self.embedding_out = torch.nn.Embedding(word_size_out, emb_dim, padding_idx= 1)
    self.encoder = Encoder(num_layer, emb_dim, head, dim_expan = dim_expan, post_LN = post_LN, dropout = dropout) # num_layer, seq_len, emb_dim, head
    self.decoder = Decoder(num_layer, emb_dim, head, dim_expan = dim_expan, post_LN = post_LN, dropout = dropout) #num_layer, seq_len, emb_dim, head
    self.linear = torch.nn.Linear(emb_dim, word_size_out) #,bias = False
    #self.embedding_in.weight = self.embedding_out.weight
    self.linear.weight = self.embedding_out.weight

  def forward(self, x, y):
    src_mask, tgt_mask = get_mask(x, 1), get_mask(y, 1)
    x, y = self.embedding_in(x) * math.sqrt(self.emb_dim), self.embedding_out(y) * math.sqrt(self.emb_dim)
    x, y = self.pos_emb(x), self.pos_emb(y)
    out_enc = self.encoder(x, src_mask)
    out = self.decoder(y, out_enc, tgt_mask, src_mask)
    out = self.linear(out) 
    return out
