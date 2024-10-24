import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import nn

import models
from models import register


def entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    """Calculates the entropy loss using PyTorch."""
    flat_affinity = affinity.view(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity , dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)

    if loss_type == "softmax":
        target_probs = probs
    elif loss_type == "argmax":
        codes = torch.argmax(flat_affinity, dim=-1)
        onehots = F.one_hot(codes, num_classes=flat_affinity.shape[-1]).to(flat_affinity.dtype)
        onehots = probs - (probs - onehots).detach()
        target_probs = onehots
    else:
        raise ValueError(f"Entropy loss {loss_type} not supported")
    
    avg_probs = torch.mean(target_probs, dim=0)    
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss, sample_entropy, avg_entropy


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=True):
        self.device = parameters.device
        self.deterministic = deterministic
        self.mean, self.logvar = parameters[..., ::2], parameters[..., 1::2]
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        if self.deterministic:
            return self.mean
        else:
            x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.device)
            return x

    def kl(self):
        return 0.5 * (torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar)

    def nll(self, sample, dims=[1]):
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


@register("bottleneck")
class Bottleneck(nn.Module):
    def __init__(
        self,
        bottleneck_dim: int,
        input_dim: int,
        output_dim: int,
        token_nums: int,
        norm=None,
        regularizer=None,
        param_names=None

    ):  
        super().__init__()
        self.token_nums = token_nums
        self.param_names = param_names
        self.input_dim = input_dim
        self.output_dim = output_dim
        if bottleneck_dim > 0:
            self.bottleneck_dim = bottleneck_dim
        else:
            assert self.input_dim == self.output_dim, "input_dim and output_dim must be the same when bottleneck_dim is not specified"
            self.bottleneck_dim = self.input_dim

        if norm is not None:
            norm = norm.lower()
            if norm == 'no' or norm == 'none':
                self.norm = None
            else:
                self.norm = norm
        else:
            self.norm = None

        if regularizer is not None and 'kl' in regularizer['name'].lower() and regularizer['name'].lower() not in ['vqkl']:
            self.project_dim = self.bottleneck_dim * 2
        else:
            self.project_dim = self.bottleneck_dim

        assert isinstance(self.input_dim, int) and isinstance(self.token_nums, int), "input_dim and n_tokens must be integers"

        token_nums_total = self.token_nums

        self.param_bottnecks = None
        if self.bottleneck_dim > 0:
            self.in_linear = nn.Linear(self.input_dim, self.project_dim)
            self.out_linear = nn.Linear(self.bottleneck_dim, self.output_dim)
        else:
            self.in_linear = self.out_linear = lambda x: x

        if self.norm is None:
            self.norm_layer = lambda x: x
        elif self.norm == 'bn_bn': # normalize among the batch dimension and the token dimension
            self.norm_layer = nn.SyncBatchNorm(self.project_dim)
        elif self.norm == 'bn_b': # only normalize among the batch dimension
            assert self.token_nums is not None, "num_tokens must be specified for batch normalization"
            self.norm_layer = nn.SyncBatchNorm(self.project_dim * self.token_nums)
        elif self.norm == 'ln_d': # only normalize among the bottleneck dimension 
            self.norm_layer = nn.LayerNorm(self.project_dim)
        elif self.norm == 'ln_nd': # normalize among the token dimension and the bottleneck dimension
            self.norm_layer = nn.LayerNorm((self.token_nums, self.project_dim))
        elif self.norm == 'ln_d_na':
            self.norm_layer = nn.LayerNorm(self.project_dim, elementwise_affine=False)
        else:
            raise ValueError(f"Normalization type {norm} not supported")

        if regularizer is not None:
            if regularizer['name'].lower() not in ['no', 'none']:
                regularizer['args']['dim'] = self.bottleneck_dim
                regularizer['args']['token_nums'] = token_nums_total
                self.regularizer = models.make(regularizer)
            else:
                self.regularizer = None
        else:
            self.regularizer = None

    def project_in(self, x):
        assert len(x.shape) == 3, "Input shape must be (batch, n_tokens, e_dim)"
        z = self.in_linear(x)
        if self.norm is None:
            pass
        else:
            with torch.autocast(device_type='cuda', enabled=False):
                z = z.float()
                if self.norm == 'bn_bn':
                    z = rearrange(z, 'b n d -> b d n')
                    z = self.norm_layer(z)
                    z = rearrange(z, 'b d n -> b n d')
                elif self.norm == 'bn_b':
                    z = rearrange(z, 'b n d -> b (n d)')
                    z = self.norm_layer(z)
                    z = rearrange(z, 'b (n d) -> b n d', n=x.shape[1])
                elif self.norm == 'ln_nd':
                    z = self.norm_layer(z)
                elif self.norm == 'ln_d':
                    z = self.norm_layer(z)
        return z

    def project_out(self, z_cat):
        z = self.out_linear(z_cat)
        return z

    def decode(self, bottleneck_rep):
        regularized_z = self.regularizer.decode(bottleneck_rep)
        return self.project_out(regularized_z)

    def forward(self, x):  
        input_norm_first = torch.norm(x[:, 0, :], dim=-1).mean().item()
        input_norm_last = torch.norm(x[:, -1, :], dim=-1).mean().item()
        z = self.project_in(x)
        projected_z = z
        if self.regularizer is not None:
            regularized_output = self.regularizer(z)
        else:
            regularized_output = {'regularized_z': z}
        x_hat = self.project_out(regularized_output['regularized_z'])
        bottleneck_rep = regularized_output.pop('bottleneck_rep')
        return {
            'output': x_hat,
            'bottleneck_rep': bottleneck_rep,
            'projected_z': projected_z,
            'input_norm_first': input_norm_first,
            'input_norm_last': input_norm_last,
            **regularized_output,
        }


class AbstractRegularizer(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, z):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError


@register("vq")
class SimpleVectorQuantizer(AbstractRegularizer):
    def __init__(
        self,
        dim,
        codebook_size,
        commitment_loss_weight=0.25,
        entropy_loss_weight=0.0,
        entropy_loss_temperature=0.01,
        l2_normalized=False,
        same_index_shape=True,
        stochastic=False,
        stochastic_temperature=1.0,
        codebook_loss_weight=1.0,
        **kwargs,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta = commitment_loss_weight
        self.codebook_loss_weight = codebook_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        assert isinstance(l2_normalized, bool)
        self.l2_normalized = l2_normalized
        self.stochastic = stochastic
        self.eval_deterministic = False
        self.default_stochastic_temperature = stochastic_temperature
        
        if self.stochastic:
            if stochastic_temperature > 0: # fixed temperature
                self.stochastic_temperature_inv = 1 / stochastic_temperature
            else: # set stochastic_temperature < 0 to use learnable temperature
                self.stochastic_temperature_inv = nn.Parameter(torch.tensor(10.0))

        self.embedding = nn.Embedding(self.codebook_size, self.dim)
        nn.init.kaiming_uniform_(self.embedding.weight)

        self.same_index_shape = same_index_shape
        self.entropy_loss_temperature = entropy_loss_temperature


    def set_eval_deterministic(self, deterministic=True):
        self.eval_deterministic = deterministic


    def set_stochastic_temperature(self, temperature):
        self.stochastic_temperature_inv = 1 / temperature


    @torch.autocast(device_type='cuda', enabled=False)
    def get_emb(self):
        if self.l2_normalized:
            emb = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            emb = self.embedding.weight
        assert emb.dtype == torch.float32, f"Embedding weight dtype is {emb.dtype}, expected float32"
        return emb


    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, z):
        z = z.float()
        assert len(z.shape) == 3, "Input shape must be (batch, n_tokens, e_dim)"
        if self.l2_normalized:
            z = F.normalize(z, p=2, dim=-1)

        emb = self.get_emb()
        z_flattened = rearrange(z, 'b n d -> (b n) d')

        if self.stochastic:
            # sample the softmaxed cosine similarity
            assert self.l2_normalized, "Stochastic sampling requires l2 normalization"
            cos_sim = torch.einsum("bd,nd->bn", z_flattened, emb)
            probs = F.softmax(cos_sim * self.stochastic_temperature_inv, dim=-1)
            if self.eval_deterministic and not self.training:
                q_indices = torch.argmax(probs, dim=-1)
            else:
                q_indices = torch.multinomial(probs, 1).squeeze(-1)
        else:
            d = (
                torch.sum(z_flattened**2, dim=1, keepdim=True)
                + torch.sum(emb**2, dim=1)
                - 2
                * torch.einsum(
                    "bd,dn->bn", z_flattened, rearrange(emb, "n d -> d n")
                )
            )
            q_indices = torch.argmin(d, dim=1)

        quantized = F.embedding(q_indices, emb, self.embedding.padding_idx, self.embedding.max_norm,
            self.embedding.norm_type, self.embedding.scale_grad_by_freq, self.embedding.sparse).view(z.shape)  # (b, n, d)
        
        loss_commit = (quantized.detach() - z) ** 2
        loss_codebook = (quantized - z.detach()) ** 2
        loss_commit = loss_commit.mean()
        loss_codebook = loss_codebook.mean()

        if self.entropy_loss_weight > 0:
            loss_entropy, sample_entropy, avg_entropy = entropy_loss(-d, temperature=self.entropy_loss_temperature)
        else:
            loss_entropy = sample_entropy = avg_entropy = torch.tensor(0.0, device=z.device, dtype=z.dtype)
        loss = self.beta * loss_commit + self.codebook_loss_weight * loss_codebook + self.entropy_loss_weight * loss_entropy

        # preserve gradients
        quantized = z + (quantized - z).detach()

        if self.same_index_shape:
            q_indices = q_indices.reshape(quantized.shape[0], quantized.shape[1])

        return_dict = {
            'unregularized_z': z, # but l2 normalized if l2_normalized=True
            'emb': emb, # but l2 normalized if l2_normalized=True
            'regularized_z': quantized,
            'bottleneck_rep': q_indices,
            'loss_q': loss,
            'loss_commit': loss_commit,
            'loss_codebook': loss_codebook,
            'loss_entropy': loss_entropy,
            'per_sample_entropy': sample_entropy,
            'codebook_entropy': avg_entropy
        }
        return return_dict
    

    def get_codebook_entry(self, indices, shape=None):
        # shape specifying (batch, height, width, channel)
        indices_shape = indices.shape
        indices_flatten = rearrange(indices, '... -> (...)')

        # get quantized latent vectors
        z_q = self.embedding(indices_flatten)
        if self.l2_normalized:
            z_q = F.normalize(z_q, p=2, dim=-1)

        if shape is not None:
            z_q = z_q.reshape(shape)
        else:
            z_q = z_q.reshape([*indices_shape, self.dim])
        return z_q

    def decode(self, indices):
        return self.get_codebook_entry(indices)
    

@register("skl") # kl divergence with summed kl loss, same as ldm's
class SummedKLDivergenceRegularizer(AbstractRegularizer):
    def __init__(
        self,
        dim,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim


    def forward(self, z):
        assert len(z.shape) == 3, "Input shape must be (batch, n_tokens, dim)"
        assert z.shape[-1] == self.dim * 2, "Input shape must be (batch, n_tokens, 2 * dim)"

        z_dist = DiagonalGaussianDistribution(z, deterministic=False)
        z_sampled = z_dist.sample()
        loss_kl = z_dist.kl()
        loss_kl = loss_kl.sum(dim=list(range(1, loss_kl.ndim))).mean()
        z_bottleneck = z_dist.mode()

        return {
            'regularized_z': z_sampled,
            'bottleneck_rep': z_bottleneck, 
            'dist': z_dist,
            'loss_kl': loss_kl
        }


    def decode(self, z_bottleneck):
        return z_bottleneck






