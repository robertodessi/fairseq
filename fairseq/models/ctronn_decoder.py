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
from .ctronn_attention import *

from einops import rearrange, reduce, asnumpy, parse_shape
import numpy as np


class CtronnDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, n_heads, args):
        super().__init__()
        self.n_heads = n_heads

        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.dropout
        self.attn_type = args.decoder_attention_type

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args
        )

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def _build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True)

    def build_self_attention(self, embed_dim, args):
        if self.attn_type == 'relative':
            assert args.decoder_attention_span > 0
            return MultiheadRelativeAttention(embed_dim, 
                self.n_heads,
                span=args.decoder_attention_span,
                dropout=args.attention_dropout)
        elif self.attn_type == 'plain':
            return MultiheadAttentionNew(
                embed_dim,
                self.n_heads,
                dropout=args.attention_dropout
            )
        elif self.attn_type == 'cnn':
            assert False
        elif self.attn_type == 'cnn-alike':
            return CnnAlikeAttention(
                embed_dim,
                self.n_heads,
                dropout=args.attention_dropout)
        elif self.attn_type == 'masking-cnn':
            return CnnMaskingAttention(
                embed_dim,
                self.n_heads,
                span=args.decoder_attention_span,
                dropout=args.attention_dropout)
        else:
            assert False

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True
        )

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        residual = x

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                static_kv=True
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.encoder_attn_layer_norm(x)

        residual = x

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        pass
