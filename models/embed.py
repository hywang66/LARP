import math
from typing import Callable, Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class VideoPatchEmbed(PatchEmbed):
    def __init__(self, *args, **kwargs):
        assert 'frame_num' in kwargs
        frame_num = kwargs.pop('frame_num')
        super().__init__(*args, **kwargs)
        
        self.num_patches_per_frame = self.num_spatial_patches = self.num_patches
        self.num_temporal_patches = frame_num


    def forward(self, x):
        b = x.size(0)
        x = einops.rearrange(x, 'b c t h w -> (b t) c h w')
        x = super().forward(x)
        if self.flatten: # x is (b t) n c
            x = einops.rearrange(x, '(b t) n c -> b (t n) c', b=b)
        else: # x is (b t) c h w
            x = einops.rearrange(x, '(b t) c h w -> b c t h w', b=b)
        return x


class PatchEmbed3D(nn.Module):
    """ 3D Video to Patch Embedding
    """
    output_fmt: str
    dynamic_vid_pad: torch.jit.Final[bool]

    def __init__(
            self,
            spatial_vid_size: Optional[int] = 224,
            temporal_vid_size: Optional[int] = 8,
            spatial_patch_size: int = 16,
            temporal_patch_size: int = 4,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_vid_size: bool = True,
            dynamic_vid_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = (temporal_patch_size, spatial_patch_size, spatial_patch_size)
        if spatial_vid_size is not None and temporal_vid_size is not None:
            self.vid_size = (temporal_vid_size, spatial_vid_size, spatial_vid_size)
            self.grid_size = tuple([s // p for s, p in zip(self.vid_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
            self.num_spatial_patches = self.num_patches_per_frame = self.grid_size[1] * self.grid_size[2]
            self.num_temporal_patches = self.grid_size[0]
        else:
            self.vid_size = None
            self.grid_size = None
            self.num_patches = None
            self.num_spatial_patches = self.num_patches_per_frame = None
            self.num_temporal_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = output_fmt.lower()
        else:
            self.flatten = flatten
            self.output_fmt = 'bcthw'
        self.strict_vid_size = strict_vid_size
        self.dynamic_vid_pad = dynamic_vid_pad

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        if self.vid_size is not None:
            if self.strict_vid_size:
                _assert(T == self.vid_size[0], f"Input depth ({T}) doesn't match model ({self.vid_size[0]}).")
                _assert(H == self.vid_size[1], f"Input height ({H}) doesn't match model ({self.vid_size[1]}).")
                _assert(W == self.vid_size[2], f"Input width ({W}) doesn't match model ({self.vid_size[2]}).")
            elif not self.dynamic_vid_pad:
                _assert(
                    T % self.patch_size[0] == 0,
                    f"Input depth ({T}) should be divisible by patch size ({self.patch_size[0]})."
                )
                _assert(
                    H % self.patch_size[1] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[1]})."
                )
                _assert(
                    W % self.patch_size[2] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[2]})."
                )
        if self.dynamic_vid_pad:
            pad_d = (self.patch_size[0] - T % self.patch_size[0]) % self.patch_size[0]
            pad_h = (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1]
            pad_w = (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2]
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BND
        elif self.output_fmt != 'bcthw':
            raise NotImplementedError(f"Output format {self.output_fmt} not supported.")
        x = self.norm(x)
        return x

def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)
    

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype=self.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LatentTokenEmbedder(nn.Module):
    """
    Embeds discrete latent tokens into vector representations. Also handles token dropout for classifier-free guidance.
    """
    def __init__(self, codebook_size, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(codebook_size + use_cfg_embedding, hidden_size)
        self.num_classes = codebook_size
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        labels: (b, n) LongTensor of latent token indices.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids[:, None], self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        # labels: (b, n) LongTensor of latent token indices.
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    

class LatentContEmbedder(nn.Module):
    """
    Embeds continuous latent codes into vector representations.
    """
    def __init__(self, token_dim, hidden_szie, dropout_prob):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.token_dim = token_dim
        self.embedding_map = nn.Linear(token_dim, hidden_szie)
        if dropout_prob > 0:
            self.uncond_embed = nn.Parameter(torch.zeros(hidden_szie), requires_grad=True)

    def emb_drop(self, embs, force_drop_ids=None):
        # embs: (b, n, h) FloatTensor of continuous latent codes.
        if force_drop_ids is None:
            drop_ids = torch.rand(embs.shape[0], device=embs.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        embs = torch.where(drop_ids[:, None, None], self.uncond_embed, embs)
        return embs
    
    def forward(self, embs, train, force_drop_ids=None):
        # embs: (b, token_dim, h) FloatTensor of continuous latent codes.
        embs = self.embedding_map(embs)
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            embs = self.emb_drop(embs, force_drop_ids)
        return embs


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        # replace all negative labels with the last class (unconditional class)
        labels = torch.where(labels < 0, self.num_classes, labels)
        embeddings = self.embedding_table(labels)
        return embeddings





#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
    
def get_3d_sincos_pos_embed(embed_dim, grid_size, frame_num):
    emb_2d = get_2d_sincos_pos_embed(embed_dim, grid_size)
    grid_1d = np.arange(frame_num, dtype=np.float32)
    emb_1d = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_1d)
    emb_2d_in_thw = emb_2d.reshape([1, grid_size, grid_size, embed_dim])
    emb_1d_in_thw = emb_1d.reshape([frame_num, 1, 1, embed_dim])
    emb_3d_in_thw = emb_2d_in_thw + emb_1d_in_thw
    emb_3d = emb_3d_in_thw.reshape([-1, embed_dim])
    return emb_3d



# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, scale_factor=10000):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    scale_factor: the base for the scaling factor, default is 10000
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / scale_factor**omega  # Parameterized scaling factor (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_circular_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    # Normalize positions to [0, 2*pi]
    max_pos = len(pos)
    assert embed_dim % 4 == 0
    pos_normalized = (pos / max_pos) * 2 * np.pi

    # Calculate sin and cos of normalized positions
    sin_pos = np.sin(pos_normalized) * (max_pos / 2)
    cos_pos = np.cos(pos_normalized) * (max_pos / 2)

    # Encode sin and cos values using get_1d_sincos_pos_embed_from_grid
    sin_embed = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, sin_pos)
    cos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, cos_pos)

    # Concatenate the sin and cos embeddings
    circular_pos_embed = np.concatenate([sin_embed, cos_embed], axis=1)
    return circular_pos_embed


