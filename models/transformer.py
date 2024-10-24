import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as BlockTimm

from models import register


@register('transformer_encoder_fused')
class TransformerEncoderFused(nn.Module):
    def __init__(self, dim, depth, n_head, head_dim, ff_dim=None, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        assert ff_dim is None
        assert dim == head_dim * n_head

        self.blocks = nn.Sequential(
            *[
                BlockTimm(
                    dim=dim,
                    num_heads=n_head,
                    mlp_ratio=4,
                    qkv_bias=False,
                    proj_drop=dropout,
                    attn_drop=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        return self.blocks(x)


@register('transformer_encoder_parallel')
class TransformerEncoderParallel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        n_head,
        head_dim,
        ff_dim=None,
        dropout=0.0
    ):
        super().__init__()
        self.is_encoder_decoder = True
        assert ff_dim is None
        assert dim == head_dim * n_head
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                BlockTimm(
                    dim=dim,
                    num_heads=n_head,
                    mlp_ratio=4,
                    qkv_bias=False,
                    proj_drop=dropout,
                    attn_drop=dropout,
                )
            )

    def forward(self, context, query):
        query_length = query.size(1)
        h = torch.cat([context, query], dim=1)

        for block in self.blocks:
            h = block(h)

        h = h[:, -query_length:, :]
        return h
