import torch
import os
import math
import numpy as np
import torch.nn.functional as F

def get_mask(x, pad_idx):
  mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)
  return mask


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
  def __init__(self, emb_dim, head, mask = False, dropout = 0):
    super(Attention, self).__init__()
    self.emb_dim = emb_dim
    self.head = head
    self.mask = mask
    self.softmax = torch.nn.Softmax(dim = -1)
    self.dropout = torch.nn.Dropout(p = dropout)

  #sent k.T in (transpose k before sent in forward)
  def forward(self, q, k, v, mask_pad):
    qk = torch.matmul(q, k) / math.sqrt(self.emb_dim//self.head)
    qk = qk.masked_fill(mask_pad == 0, -1e20)
    if self.mask == True:
      qk = torch.tril(qk)
      qk = qk.masked_fill(qk == 0, -1e20)
    att_w = self.dropout(self.softmax(qk))
    out = torch.matmul(att_w, v)
    return out
  
  
class Multi_Head_ATT(torch.nn.Module):
    def __init__(self, seq_len, emb_dim, multi_head = 1, mask = False, post_LN = True, dropout = 0):
      super(Multi_Head_ATT,self).__init__()
      self.head = multi_head
      self.emb_dim = emb_dim
      self.seq_len = seq_len
      self.post_LN = post_LN
      self.q_att = torch.nn.Linear(emb_dim, emb_dim, bias = False) 
      self.k_att = torch.nn.Linear(emb_dim, emb_dim, bias = False) 
      self.v_att = torch.nn.Linear(emb_dim, emb_dim, bias = False) 
      self.attention = Attention(emb_dim, multi_head, mask = mask, dropout = dropout)
      self.LN = torch.nn.LayerNorm(emb_dim, eps = 1e-6)
      self.WO = torch.nn.Linear(emb_dim, emb_dim)
      self.dropout = torch.nn.Dropout(p = dropout)

    def forward(self, q,k,v, mask_pad):
      res = q
      if self.post_LN == False:
        q, k, v = self.LN(q), self.LN(k), self.LN(v)
      if self.head == 1:
        q, k, v = self.q_att(q), self.k_att(k).permute(0,2,1), self.v_att(v)
        out = self.attention(q, k, v, mask_pad)
      else:
        # (b_s, seq_len, head, emb//head) > (b_s, head, seq_len, emb_dim//head)
        q = self.q_att(q).view(-1,self.seq_len,self.head,self.emb_dim//self.head).permute(0,2,1,3)
        k = self.k_att(k).view(-1,self.seq_len,self.head,self.emb_dim//self.head).permute(0,2,1,3).permute(0,1,3,2)
        v = self.v_att(v).view(-1,self.seq_len,self.head,self.emb_dim//self.head).permute(0,2,1,3)
        out = self.attention(q, k, v, mask_pad).permute(0,2,1,3).contiguous().view(-1,self.seq_len,self.emb_dim)
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
  def __init__(self, num_layer, seq_len, emb_dim, head, dim_expan = 4, post_LN = True, dropout = 0):
    super(Encoder, self).__init__()
    self.num_layer = num_layer
    self.attention = Multi_Head_ATT(seq_len, emb_dim, multi_head = head, dropout = dropout)
    self.FF = Feed_Forward(emb_dim, dim_expan = dim_expan, dropout = dropout)
    self.connect1 = torch.nn.ModuleList([
                                         Multi_Head_ATT(seq_len, emb_dim, multi_head = head, post_LN = post_LN, dropout = dropout) for i in range(num_layer - 1)
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
  def __init__(self,num_layer, seq_len, emb_dim, head, dim_expan = 4, post_LN = True, dropout = 0):
    super(Decoder,self).__init__()
    self.num_layer = num_layer
    self.attention = Multi_Head_ATT(seq_len, emb_dim, multi_head= head, mask = True, dropout = dropout)
    self.cross_att = Multi_Head_ATT(seq_len, emb_dim, multi_head= head, dropout = dropout)
    self.FF = Feed_Forward(emb_dim, dim_expan = dim_expan, dropout = dropout)
    self.connect_1 = torch.nn.ModuleList([
                                          Multi_Head_ATT(seq_len, emb_dim, multi_head= head, mask = True, post_LN = post_LN, dropout = dropout) for i in range(self.num_layer - 1)
                                          ])
    self.connect_2 = torch.nn.ModuleList([
                                          Multi_Head_ATT(seq_len, emb_dim, multi_head= head, post_LN = post_LN, dropout = dropout) for i in range(self.num_layer -1)
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
    self.encoder = Encoder(num_layer, seq_len, emb_dim, head, dim_expan = dim_expan, post_LN = post_LN, dropout = dropout) # num_layer, seq_len, emb_dim, head
    self.decoder = Decoder(num_layer, seq_len, emb_dim, head, dim_expan = dim_expan, post_LN = post_LN, dropout = dropout) #num_layer, seq_len, emb_dim, head
    self.linear = torch.nn.Linear(emb_dim, word_size_out) #,bias = False
    #self.embedding_in.weight = self.embedding_out.weight
    self.linear.weight = self.embedding_out.weight

  def forward(self, x, y):
    src_mask, tgt_mask = get_mask(x, 1), get_mask(y, 1)
    x, y = self.embedding_in(x) * math.sqrt(self.emb_dim), self.embedding_out(y) * math.sqrt(self.emb_dim)
    x, y = self.pos_emb(x), self.pos_emb(y)
    out_enc = self.encoder(x, src_mask)
    out = self.decoder(y, out_enc, tgt_mask, src_mask)# + tgt_mask)
    out = self.linear(out) #* math.sqrt(self.emb_dim)
    return out
