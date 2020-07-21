# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm
from torch import Tensor

from einops import rearrange, reduce
import numpy as np
from models.ctronn_attention import *


class CtronnEncoderLayer(nn.Module):
    def __init__(self, args, attn_type, n_heads):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim

        self.attn_type = attn_type
        self.n_heads = n_heads

        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.fc1 = nn.Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.encoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.glu = getattr(args, "encoder_glu", False)
       
    def build_self_attention(self, embed_dim, args):
        if self.attn_type == 'relative':
            assert args.encoder_attention_span > 0
            return MultiheadRelativeAttention(embed_dim, 
                self.n_heads,
                span=args.encoder_attention_span,
                dropout=args.attention_dropout)
        elif self.attn_type == 'plain':
            return MultiheadAttentionNew(
                embed_dim,
                self.n_heads,
                dropout=args.attention_dropout
            )
        elif self.attn_type == 'cnn':
            return CnnAttention(
                in_dim=embed_dim,
                out_dim=(embed_dim if not args.encoder_glu else 2 * embed_dim),
                n_head=self.n_heads,
                dropout=args.attention_dropout,
                use_layer_norm=True
            )
        elif self.attn_type == 'cnn-alike':
            return CnnAlikeAttention(
                embed_dim,
                self.n_heads,
                dropout=args.attention_dropout
            )
        elif self.attn_type == 'masking-cnn':
            return CnnMaskingAttention(
                embed_dim,
                self.n_heads,
                #span=args.encoder_attention_span,
                dropout=args.attention_dropout)
        else:
            assert False

    def upgrade_state_dict_named(self, state_dict, name):
        pass

    def forward(self, x, 
                    encoder_padding_mask,
                    attn_mask: Optional[Tensor] = None,
                    additive_attention = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        #if attn_mask is not None:
        #    attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -np.inf)

        residual = x

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask
        )
        if not (self.attn_type == 'cnn' and self.glu):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x

            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
        else:
            x = F.glu(x, dim=-1)
            x = x + residual
        x = self.final_layer_norm(x)
        return x
































































































































































































