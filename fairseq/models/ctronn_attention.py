# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.incremental_decoding_utils import with_incremental_state
from torch import Tensor

from einops import rearrange, reduce, asnumpy, parse_shape
import numpy as np
import math
from torch.nn import Parameter
from scipy.linalg import toeplitz


class RelativeBias(nn.Module):
    def __init__(self,
                 n_heads,
                 span):
        super().__init__()

        self.n_heads = n_heads
        self.embedding = nn.Embedding(2 * span + 1, n_heads)
        self.span = span

        self._allocate_difference_matrix(span)

    def _allocate_difference_matrix(self, l):
        t = toeplitz(range(l))
        self.relative_position = torch.from_numpy(np.triu(t) - np.tril(t))
        self.relative_position.clamp_(min=-self.span, max=+self.span)
        self.relative_position += self.span

    def forward(self, attn):
        # attn is (h b l t)
        l = attn.size(2)

        if l > self.relative_position.size(0):
            self._allocate_difference_matrix(l)
        self.relative_position = self.relative_position.to(attn.device)

        relative_positons = rearrange(self.relative_position[:l, :l], "t l -> (t l)")
        embedded = self.embedding(relative_positons)
        relative_embeddings = rearrange(embedded, "(t l) h -> h () t l", l=l).expand_as(attn)
        return relative_embeddings + attn


class MultiheadRelativeAttention(nn.Module):
    """
    Implements T5-like relative attn
    """
    def __init__(self, 
                 d_model, 
                 n_heads,
                 span,
                 d_k=None, 
                 d_v=None, 
                 dropout=0.1):
        super().__init__()
        self.n_head = n_heads

        d_k = d_k if d_k else d_model
        d_v = d_v if d_v else d_model

        self.w_qs = nn.Linear(d_model, n_heads * d_k)
        self.w_ks = nn.Linear(d_model, n_heads * d_k)
        self.w_vs = nn.Linear(d_model, n_heads * d_v)
        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        
        self.fc = nn.Linear(n_heads * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.relative_attention = RelativeBias(n_heads, span)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        residual = query

        query = rearrange(self.w_qs(query), 'l b (head k) -> head b l k', head=self.n_head)
        key = rearrange(self.w_ks(key), 't b (head k) -> head b t k', head=self.n_head)
        value = rearrange(self.w_vs(value), 't b (head v) -> head b t v', head=self.n_head)

        attn = torch.einsum('hblk,hbtk->hblt', [query, key]) / np.sqrt(query.shape[-1])

        # the only difference!
        attn = self.relative_attention(attn)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0).unsqueeze(2)
            key_padding_mask = key_padding_mask.expand_as(attn)
            attn = attn.masked_fill(key_padding_mask, -np.inf)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand_as(attn)
            attn = attn.masked_fill(attn_mask, -np.inf)

        attn = torch.softmax(attn, dim=-1)
        output = torch.einsum('hblt,hbtv->hblv', [attn, value])
        output = rearrange(output, 'head b l v -> l b (head v)')
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


# plain multi-head attn in a simpler implementation
class MultiheadAttentionNew(nn.Module):
    def __init__(self, 
                 d_model, 
                 n_head, 
                 d_k=None, 
                 d_v=None, 
                 dropout=0.1):
        super().__init__()
        self.n_head = n_head

        d_k = d_k if d_k else d_model
        d_v = d_v if d_v else d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        residual = query

        query = rearrange(self.w_qs(query), 'l b (head k) -> head b l k', head=self.n_head)
        key = rearrange(self.w_ks(key), 't b (head k) -> head b t k', head=self.n_head)
        value = rearrange(self.w_vs(value), 't b (head v) -> head b t v', head=self.n_head)

        attn = torch.einsum('hblk,hbtk->hblt', [query, key]) / np.sqrt(query.shape[-1])

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0).unsqueeze(2)
            key_padding_mask = key_padding_mask.expand_as(attn)
            attn = attn.masked_fill(key_padding_mask, -np.inf)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand_as(attn)
            attn = attn.masked_fill(attn_mask, -np.inf)

        attn = torch.softmax(attn, dim=-1)
        output = torch.einsum('hblt,hbtv->hblv', [attn, value])
        output = rearrange(output, 'head b l v -> l b (head v)')
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

def ConvTBC(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    from fairseq.modules import ConvTBC
    m = ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)


class CnnAttention(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_dim,
                 n_head,
                 dropout,
                 use_layer_norm): # kernel size = n_head
        super().__init__()
        self.n_head = n_head

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(out_dim) if use_layer_norm else None

        # can crush if padding is mis-sized?
        self.conv = ConvTBC(in_dim, out_dim, n_head, padding=(n_head - 1) // 2, dropout=dropout)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        # key padding is B L
        x = query # L B D
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.permute(1, 0).unsqueeze(-1), 0)
        assert attn_mask is None

        #residual = query
        output = self.conv(query)
        output = self.dropout(output) #+ residual
        if self.layer_norm:
            output = self.layer_norm(output)
        return output, None


class CnnBias(nn.Module):
    def __init__(self,
                 n_heads):
        super().__init__()

        self.n_heads = n_heads
        weight = torch.diag(torch.tensor([1 for _ in range(n_heads)])).float() * 100
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        
        self.span = (n_heads - 1) // 2


        # init length at 10
        self._allocate_difference_matrix(10)

    def _allocate_difference_matrix(self, length):
        t = toeplitz(range(length))
        self.relative_position = torch.from_numpy(np.triu(t) - np.tril(t))
        self.relative_position.clamp_(min=-self.span, max=+self.span)
        self.relative_position += self.span

    def forward(self, attn):
        # attn is (h b l t)
        l = attn.size(2)

        if l > self.relative_position.size(0):
            self._allocate_difference_matrix(l)
        self.relative_position = self.relative_position.to(attn.device)

        relative_positons = rearrange(self.relative_position[:l, :l], "t l -> (t l)")
        embedded = self.embedding(relative_positons)
        relative_embeddings = rearrange(embedded, "(t l) h -> h () t l", l=l).expand_as(attn)
        return relative_embeddings# + attn


class CnnAlikeAttention(nn.Module):
    def __init__(self, 
                 d_model, 
                 n_heads,
                 span=None,
                 d_k=None, 
                 d_v=None, 
                 dropout=0.1):
        super().__init__()
        self.n_head = n_heads
        span = (n_heads - 1) // 2 if not span else span

        d_k = d_k if d_k else d_model
        d_v = d_v if d_v else d_model

        self.w_qs = nn.Linear(d_model, n_heads * d_k)
        self.w_ks = nn.Linear(d_model, n_heads * d_k)
        self.w_vs = nn.Linear(d_model, n_heads * d_v)
        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        
        self.fc = nn.Linear(n_heads * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.relative_attention = CnnBias(n_heads)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        residual = query

        query = rearrange(self.w_qs(query), 'l b (head k) -> head b l k', head=self.n_head)
        key = rearrange(self.w_ks(key), 't b (head k) -> head b t k', head=self.n_head)
        value = rearrange(self.w_vs(value), 't b (head v) -> head b t v', head=self.n_head)

        attn = torch.einsum('hblk,hbtk->hblt', [query, key]) / np.sqrt(query.shape[-1])

        # the only difference!
        attn = self.relative_attention(attn)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0).unsqueeze(2)
            key_padding_mask = key_padding_mask.expand_as(attn)
            attn = attn.masked_fill(key_padding_mask, -np.inf)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand_as(attn)
            attn = attn.masked_fill(attn_mask, -np.inf)

        attn = torch.softmax(attn, dim=-1)

        output = torch.einsum('hblt,hbtv->hblv', [attn, value])
        output = rearrange(output, 'head b l v -> l b (head v)')
        output = self.dropout(self.fc(output))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output, attn



class CnnMaskingAttention(nn.Module):
    def __init__(self, 
                 d_model, 
                 n_heads,
                 span=None,
                 d_k=None, 
                 d_v=None, 
                 dropout=0.1):

        super().__init__()
        self.n_head = n_heads
        self.span = (n_heads - 1) // 2 if not span else span

        d_k = d_k if d_k else d_model
        d_v = d_v if d_v else d_model

        self.w_qs = nn.Linear(d_model, n_heads * d_k)
        self.w_ks = nn.Linear(d_model, n_heads * d_k)
        self.w_vs = nn.Linear(d_model, n_heads * d_v)
        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        
        self.fc = nn.Linear(n_heads * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # some large-enough initial value
        self._allocate_mask_matrix(20)

    def _allocate_mask_matrix(self, l):
        t = toeplitz(range(l))
        relative_position = torch.from_numpy(np.triu(t) - np.tril(t))

        self.cached_mask = torch.ones(l, l, dtype=torch.bool)
        self.cached_mask.masked_fill_((relative_position <= self.span) & (relative_position >= -self.span), False)

    def get_cnn_additive_mask(self, attn):
        # attn hblt
        l = attn.size(2)
        t = attn.size(3)
        assert l == t

        if l > self.cached_mask.size(0):
            self._allocate_mask_matrix(l)
        self.cached_mask = self.cached_mask.to(attn.device)

        mask = rearrange(self.cached_mask[:l, :l], 'l t -> () () l t').expand_as(attn)
        additive = torch.zeros_like(mask).masked_fill(mask, -np.inf)
        return additive

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        residual = query

        query = rearrange(self.w_qs(query), 'l b (head k) -> head b l k', head=self.n_head)
        key = rearrange(self.w_ks(key), 't b (head k) -> head b t k', head=self.n_head)
        value = rearrange(self.w_vs(value), 't b (head v) -> head b t v', head=self.n_head)

        attn = torch.einsum('hblk,hbtk->hblt', [query, key]) / np.sqrt(query.shape[-1])

        cnn_additive_mask = self.get_cnn_additive_mask(attn)
        attn = attn + cnn_additive_mask

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0).unsqueeze(2)
            key_padding_mask = key_padding_mask.expand_as(attn)
            attn = attn.masked_fill(key_padding_mask, -np.inf)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand_as(attn)
            attn = attn.masked_fill(attn_mask, -np.inf)

        attn = torch.softmax(attn, dim=-1)
        assert not torch.isnan(attn).any()
        assert not torch.isinf(attn).any()

        output = torch.einsum('hblt,hbtv->hblv', [attn, value])
        output = rearrange(output, 'head b l v -> l b (head v)')
        output = self.dropout(self.fc(output))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output, attn




class _CnnAlikeAttention(nn.Module):
    def __init__(self, 
                 d_model, 
                 n_head, 
                 d_k=None, 
                 d_v=None, 
                 dropout=0.1):
        super().__init__()
        self.n_head = n_head

        d_k = d_k if d_k else d_model
        d_v = d_v if d_v else d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def apply_cnn_bias(self, attn):
        heads, b, l_query, l_key = attn.size()

        cnn_attn = torch.zeros(heads, l_query, l_key)

        for i in range(l_query):
            for h in range(heads):
                h_relative = h - (heads - 1) // 2
                if l_key > i + h_relative >= 0:
                    j = i + h_relative
                    cnn_attn[h, i, j] = 1
                else: 
                    cnn_attn[h, i, -1] = 1

        cnn_attn = cnn_attn.to(attn).unsqueeze(1).expand_as(attn)

        #return attn + cnn_attn
        return (cnn_attn - 0.5) * 1e3
 
    def add_cnn_padding(self, m):
        _, bsz, rest = m.size()
        padding = torch.zeros(1, bsz, rest).to(m)
        return torch.cat([m, padding])

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        residual = query
        bsz = query.size(1)
        heads = self.n_head

        query = rearrange(self.w_qs(query), 'l b (head k) -> head b l k', head=heads)

        padding_size = (self.n_head - 1) // 2

        key = self.w_ks(key)
        key = self.add_cnn_padding(key)
        key = rearrange(key, 't b (head k) -> head b t k', head=heads)

        value = self.w_vs(value)
        value = self.add_cnn_padding(value)
        value = rearrange(value, 't b (head v) -> head b t v', head=heads)

        attn = torch.einsum('hblk,hbtk->hblt', [query, key]) / np.sqrt(query.shape[-1])
        attn = self.apply_cnn_bias(attn)

        if key_padding_mask is not None and key_padding_mask.any():
            # padding on the right
            padding = torch.zeros(bsz, 1).to(key_padding_mask)
            key_padding_mask = torch.cat([key_padding_mask, padding], dim=1)
            key_padding_mask = key_padding_mask.unsqueeze(0).unsqueeze(2)
            key_padding_mask = key_padding_mask.expand_as(attn)
            attn = attn.masked_fill(key_padding_mask, -np.inf)

        if attn_mask is not None:
            assert False, 'not yet handled'
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand_as(attn)
            attn = attn.masked_fill(attn_mask, -np.inf)

        attn = torch.softmax(attn, dim=-1)

        output = torch.einsum('hblt,hbtv->hblv', [attn, value])
        output = rearrange(output, 'head b l v -> l b (head v)')
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn
