from itertools import chain

import lpips
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.parametrizations import spectral_norm

from models import register
from models.transformer import TransformerEncoderFused

from .embed import PatchEmbed3D, VideoPatchEmbed, get_3d_sincos_pos_embed


def lecam_reg(real_pred, fake_pred, ema_real_pred, ema_fake_pred):
    """Lecam loss for data-efficient and stable GAN training.
    
    Described in https://arxiv.org/abs/2104.03310
    
    Args:
      real_pred: Prediction (scalar) for the real samples.
      fake_pred: Prediction for the fake samples.
      ema_real_pred: EMA prediction (scalar) for the real samples.
      ema_fake_pred: EMA prediction for the fake samples.
    
    Returns:
      Lecam regularization loss (scalar).
    """
    assert real_pred.ndim == 0 and ema_fake_pred.ndim == 0
    lecam_loss = torch.mean(torch.pow(torch.relu(real_pred - ema_fake_pred), 2))
    lecam_loss = lecam_loss + torch.mean(torch.pow(torch.relu(ema_real_pred - fake_pred), 2))
    return lecam_loss


def r1_gradient_penalty(discriminator, real_video, penalty_cost=1.0):
    
    real_video = real_video.detach().clone().requires_grad_(True)
    out = discriminator(real_video)

    # Compute gradients with respect to the inputs
    with torch.amp.autocast(device_type='cuda', enabled=False):
        out = out.float()
        gradients = autograd.grad(
            outputs=out,
            inputs=real_video,
            grad_outputs=torch.ones(out.size(), device=real_video.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(real_video.size(0), -1)
        gradient_penalty = torch.mean(torch.sum(gradients ** 2, dim=1)) * penalty_cost
    return out, gradient_penalty


def apply_spectral_norm(module: nn.Module):
    for name, layer in module.named_children():
        if isinstance(layer, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            setattr(module, name, spectral_norm(layer))
        else:
            apply_spectral_norm(layer)

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def hinge_g_loss(logits_fake):
    g_loss = -torch.mean(logits_fake)
    return g_loss

def ns_d_loss(logits_real, logits_fake):
    real_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
    fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
    d_loss = real_loss + fake_loss
    return d_loss

def ns_d_loss_single_side_smooth(logits_real, logits_fake):
    real_target =  torch.ones_like(logits_real) - torch.randn_like(logits_real).abs() * 0.15
    real_target.clamp_min_(0.7)

    fake_target = torch.randn_like(logits_fake).abs() * 0.15
    fake_target.clamp_max_(0.3)

    real_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_real, real_target)
    fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_fake, fake_target)
    d_loss = real_loss + fake_loss
    return d_loss

def ns_g_loss(logits_fake):
    g_loss = -torch.mean(F.logsigmoid(logits_fake))
    return g_loss

def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight

def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use

def l1(x, y):
    return torch.abs(x - y)

def l2(x, y):
    return torch.pow((x - y), 2)


class TransformerDiscriminator(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_heads,
        n_layers,
        input_size,
        temporal_patch_size,
        patch_size,
        in_channels,
        frame_num=16
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_heads
        self.n_layers = n_layers
        self.input_size = input_size
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.frame_num = frame_num
        
        if temporal_patch_size == 1:
            self.x_embedder = VideoPatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True, frame_num=frame_num)
        else:
            assert temporal_patch_size > 1
            self.x_embedder = PatchEmbed3D(input_size, frame_num, patch_size, temporal_patch_size, in_channels, hidden_size, bias=True)
        
        self.token_t = self.x_embedder.num_temporal_patches
        self.token_h = self.token_w = int(self.x_embedder.num_spatial_patches ** 0.5)
        self.video_token_num = video_token_num = self.x_embedder.num_spatial_patches * self.token_t

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.register_buffer('encoder_pos_embed', torch.zeros(1, video_token_num, hidden_size))
        self.get_encoder_pos_embed = lambda: self.encoder_pos_embed
        
        self.transformer_encoder = TransformerEncoderFused(
            dim=hidden_size,
            depth=n_layers,
            n_head=n_heads,
            head_dim=hidden_size // n_heads
        )

        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.fc = nn.Linear(hidden_size, 1)

        self.initialize_weights()


    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        encoder_pos_embed = get_3d_sincos_pos_embed(self.hidden_size, self.token_h, self.token_t)
        self.encoder_pos_embed.data.copy_(torch.from_numpy(encoder_pos_embed).float().reshape_as(self.encoder_pos_embed))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize cls_token using uniform distribution:
        nn.init.xavier_uniform_(self.cls_token)


    def forward(self, x):
        '''
        x: (b, c, t, h, w)
        '''
        b, c, t, h, w = x.shape
        x = self.x_embedder(x) + self.get_encoder_pos_embed() # (b, n, d)
        cls_tokens = self.cls_token.expand(b, -1, -1) # (b, 1, d)
        x = torch.cat((cls_tokens, x), dim=1) # (b, n+1, d)
        z = self.transformer_encoder(x) # (b, n+1, d)
        z_cls = z[:, 0] # (b, d)
        z_cls = self.norm_final(z_cls)
        out = self.fc(z_cls) # (b, 1)
        return out


@register('lpips_disc_loss')
class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        disc_self_start=None,
        pixelloss_weight=1.0,
        disc_type='transformer',
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        disc_loss="ns_smooth",
        disc_tran_hidden_size=256,
        disc_tran_n_heads=8,
        disc_tran_n_layers=6,
        disc_tran_temporal_patch_size=1,
        disc_tran_patch_size=16,
        frame_num=16,
        perceptual_loss="lpips",
        perceptual_fp16=False,
        pixel_loss="l1",
        lecam_weight=0.0,
        input_spatial_size=128,
        r1_gp_weight=0.0,
        d_update_freq=1,
        d_update_loss_threshold=-1.0e6,
        spectral_norm=False,
    ):
        super().__init__()
        assert disc_loss in ["hinge", "ns", "ns_smooth"]
        assert pixel_loss in ["l1", "l2"]
        self.pixel_weight = pixelloss_weight
        if perceptual_loss == "lpips":
            self.perceptual_loss = lpips.LPIPS(net='vgg')
        else:
            raise ValueError(f"Unknown perceptual loss: >> {perceptual_loss} <<")
        if perceptual_fp16:
            self.perceptual_loss = self.perceptual_loss.to(dtype=torch.float16)
        self.set_perceptual_eval()
        self.perceptual_weight = perceptual_weight

        if pixel_loss == "l1":
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2

        self.input_spatial_size = input_spatial_size
        self.r1_gp_weight = r1_gp_weight
        self.d_update_freq = d_update_freq
        self.d_update_loss_threshold = d_update_loss_threshold


        if disc_type.lower() == 'transformer':
            self.discriminator = TransformerDiscriminator(
                hidden_size=disc_tran_hidden_size,
                n_heads=disc_tran_n_heads,
                n_layers=disc_tran_n_layers,
                input_size=input_spatial_size,
                temporal_patch_size=disc_tran_temporal_patch_size,
                patch_size=disc_tran_patch_size,
                in_channels=disc_in_channels,
                frame_num=frame_num
            )
            self.disc_type = '3d'
        else:
            raise ValueError(f"Unknown discriminator type: >> {disc_type} <<")

        if spectral_norm:
            apply_spectral_norm(self.discriminator)

        self.discriminator_iter_start = disc_start
        if disc_self_start is not None and disc_self_start >= 0:
            self.discriminator_self_start = disc_self_start
        else:
            self.discriminator_self_start = disc_start

        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
            self.g_loss = hinge_g_loss
        elif disc_loss == "ns":
            self.disc_loss = ns_d_loss
            self.g_loss = ns_g_loss
        elif disc_loss == "ns_smooth":
            self.disc_loss = ns_d_loss_single_side_smooth
            self.g_loss = ns_g_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")

        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

        self.lecam_weight = lecam_weight
        if self.lecam_weight > 0.0:
            print(f"Using LeCam regularization with weight {self.lecam_weight}.")
            self.register_buffer('lecam_ema_real', torch.tensor(0.0))
            self.register_buffer('lecam_ema_fake', torch.tensor(0.0))

    def set_perceptual_eval(self):
        self.perceptual_loss.eval()
        for param in self.perceptual_loss.parameters():
            param.requires_grad_(False)

    def trainable_requires_grad_(self, requires_grad):
        for param in self.trainable_parameters():
            param.requires_grad_(requires_grad)

    def trainable_modules(self):
        return [self.discriminator]

    def trainable_parameters(self):
        return chain(*(m.parameters() for m in self.trainable_modules()))

    def set_training_mode(self, trainable_mode, others_mode=False):
        self.train(others_mode)
        for m in self.trainable_modules():
            m.train(trainable_mode)

    @torch.no_grad()
    @torch.autocast(device_type='cuda', enabled=False)
    def update_lecam_ema(self, real, fake, decay=0.999):
        real, fake = real.float().mean(), fake.float().mean()
        self.lecam_ema_real.mul_(decay).add_(real, alpha=1 - decay)
        self.lecam_ema_fake.mul_(decay).add_(fake, alpha=1 - decay)

    def forward_perceptual(self, inputs, reconstructions): 
        input_frames = rearrange(inputs, 'b c t h w -> (b t) c h w').contiguous()
        reconstruction_frames = rearrange(reconstructions, 'b c t h w -> (b t) c h w').contiguous()
        loss_perc = self.perceptual_loss(input_frames, reconstruction_frames, normalize=True)
        return {'loss_prior': loss_perc.mean()}

    def forward(
        self,
        inputs,
        reconstructions,
        global_step,
        for_discriminator=False,
        last_layer=None,
    ):  
        input_frames = rearrange(inputs, 'b c t h w -> (b t) c h w').contiguous()
        reconstruction_frames = rearrange(reconstructions, 'b c t h w -> (b t) c h w').contiguous()

        if self.disc_type == '2d':
            input_to_disc = input_frames
            recon_to_disc = reconstruction_frames
        elif self.disc_type == '3d':
            input_to_disc = inputs
            recon_to_disc = reconstructions
        else:
            raise ValueError(f"Unknown discriminator type: >> {self.disc_type} <<")

        # now the GAN part
        if not for_discriminator:
            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            if self.pixel_weight > 0:
                rec_loss = self.pixel_loss(input_frames, reconstruction_frames)
            else:
                rec_loss = input_frames.new_zeros(1)

            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(
                    input_frames, reconstruction_frames, normalize=True
                )
            else:
                p_loss = input_frames.new_zeros(1)

            rp_loss = self.pixel_weight * rec_loss + self.perceptual_weight * p_loss

            nll_loss = rp_loss
            nll_loss = torch.mean(nll_loss)

            # generator update
            if disc_factor > 0.0:
                logits_fake = self.discriminator(recon_to_disc)
                g_loss = self.g_loss(logits_fake)
                d_weight = self.discriminator_weight
            else:
                d_weight = input_frames.new_zeros(1)
                g_loss = input_frames.new_zeros(1)

            g_loss_weight = d_weight * disc_factor
            if isinstance(g_loss_weight, torch.Tensor):
                g_loss_weight = g_loss_weight.item()

            loss = nll_loss + g_loss_weight * g_loss

            info_dict = {
                'rec_loss': rec_loss.mean().item(),
                'perceptual_loss': p_loss.mean().item(),
                'rp_loss': nll_loss.item(),
                'g_loss': g_loss.item(),
                'g_loss_weight': g_loss_weight,
            }

            return loss, info_dict

        else: # for_discriminator == True, discriminator update
            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_self_start
            )
            if disc_factor > 0.0:
                if self.training and self.r1_gp_weight > 0.0:
                    logits_real, r1_gp = r1_gradient_penalty(
                        self.discriminator, input_to_disc.contiguous(), penalty_cost=self.r1_gp_weight
                    )
                else:
                    logits_real = self.discriminator(input_to_disc.contiguous())
                    r1_gp = input_frames.new_zeros(1)
                logits_fake = self.discriminator(recon_to_disc.contiguous().detach())

                if self.lecam_weight > 0.0:
                    lecam_loss = self.lecam_weight * lecam_reg(
                        real_pred=logits_real.mean(),
                        fake_pred=logits_fake.mean(),
                        ema_real_pred=self.lecam_ema_real,
                        ema_fake_pred=self.lecam_ema_fake,
                    )
                    self.update_lecam_ema(logits_real, logits_fake)
                else:
                    lecam_loss = input_frames.new_zeros(1)

                d_loss =  self.disc_loss(logits_real, logits_fake)

                total_loss = d_loss + self.lecam_weight * lecam_loss + r1_gp

            else:
                d_loss = input_frames.new_zeros(1)
                lecam_loss = input_frames.new_zeros(1)
                total_loss = input_frames.new_zeros(1)
                logits_real = input_frames.new_zeros(1)
                logits_fake = input_frames.new_zeros(1)

            info_dict = {
                'd_total_loss': total_loss.item(), 
                'd_lecam_loss': lecam_loss.item(),
                'd_loss': d_loss.item(),
                'logits_real': logits_real.mean().item(),
                'logits_fake': logits_fake.mean().item(),
            }
            if self.r1_gp_weight > 0.0:
                info_dict['r1_gp'] = r1_gp.item()

            return total_loss, info_dict
