import math
import os
from copy import deepcopy

import einops
import imageio
import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm

import utils
from models import get_model_cls
from models.embed import LabelEmbedder
from models.norm import RMSNorm
from trainers import register

from .base_trainer import BaseTrainer
from .larp_tokenizer_trainer import optimizer_dict


def save_video(video, path):
    imageio.mimwrite(path, video.numpy(), fps=25)


@register('larp_ar_trainer')
class LARPARTrainer(BaseTrainer):
    def __init__(self, rank, cfg):
        super().__init__(rank, cfg)
        self.num_samples = cfg['ar']['num_samples']
        self.num_save_wandb = cfg['ar']['num_save_wandb']
        self.sample_batch_size = cfg['ar']['sample_batch_size']
        self.cfg_scale = cfg['ar']['cfg_scale']
        self.cfg_interval = cfg['ar']['cfg_interval']
        self.sampling_temperture = cfg['ar']['temperature']
        self.sampling_top_k = cfg['ar']['top_k']
        self.sampling_top_p = cfg['ar']['top_p']
        cfg['vae']['checkpoint'] = cfg['vae']['checkpoint'].strip("'").strip('"')

        if os.path.exists(cfg['vae']['checkpoint']):
            # load model from local checkpoint
            self.vae = (
                get_model_cls(cfg['vae']['name'])
                .from_checkpoint(cfg['vae']['checkpoint'], version=cfg['vae']['version'])
                .to(self.device)
            )
        else:
            # load model from huggingface hub
            self.vae = (
                get_model_cls(cfg['vae']['name'])
                .from_pretrained(cfg['vae']['checkpoint'])
                .to(self.device)
            )

        self.vae.eval()
        vae_eval_deterministic = cfg['vae'].get('eval_deterministic', False)
        if vae_eval_deterministic:
            self.vae.set_vq_eval_deterministic(True)
            self.log('Set VQ eval mode to deterministic. Only effective if stochastic VQ is used')

        self.log(f'Loaded VAE from {cfg["vae"]["checkpoint"]}')
        self.vae_force_fp32 = cfg['vae_force_fp32'] if 'vae_force_fp32' in cfg else False

        self.seq_length = self.vae.bottleneck_token_num
        self.cfg['model']['args']['max_seq_len'] = self.seq_length
        self.cfg['model']['args']['vocab_size'] = self.vae.codebook_size
        self.log(f'Using sequence length: {self.seq_length}')
        self.log(f'Using vocab size: {self.vae.codebook_size}')


    @staticmethod
    def get_exp_name(base_exp_name, cfg, args):
        exp_name = f"{base_exp_name}/"
        if len(cfg.vae.checkpoint) < 8:
            exp_name += cfg.vae.checkpoint + "_"

        if float(cfg.optimizer.args.lr) != 0.0001:
            exp_name += f"lr{cfg.optimizer.args.lr}_"

        if 'weight_decay' in cfg.optimizer.args:
            if cfg.optimizer.args.weight_decay != 0.0:
                exp_name += f"wd{cfg.optimizer.args.weight_decay}_"

        exp_name += f'{cfg.model.name}_'
        exp_name += f'_{args.tag}'
        return exp_name


    def make_model(self, model_spec=None, load_sd=False):
        super().make_model(model_spec, load_sd)
        vae_size_str = utils.compute_num_params(self.vae)
        self.log(f'vae size: {vae_size_str}')
        if self.enable_wandb:
            wandb.run.summary['vae_size'] = vae_size_str


    def configure_optimizers(self, config, load_sd=False):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (
            torch.nn.LayerNorm, 
            torch.nn.Embedding,
            LabelEmbedder,
            RMSNorm
        )
        for mn, m in self.orig_model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        cases = ['pos_emb', 'abs_pe']
        for case in cases:
            if hasattr(self.orig_model, case) and isinstance(getattr(self.orig_model, case), torch.nn.Parameter):
                no_decay.add(case)
 
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.orig_model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        

        # config_no_decay = deepcopy(config)
        config_no_decay = deepcopy({k: v for k, v in config.items() if k != 'sd'})
        if 'weight_decay' in config_no_decay['args']:
            del config_no_decay['args']['weight_decay']

        optimizer_groups = [
            {'params': [param_dict[pn] for pn in sorted(list(decay))], 'weight_decay': config['args']['weight_decay']},
            {'params': [param_dict[pn] for pn in sorted(list(no_decay))], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_dict[config['name']](
            optimizer_groups,
            **config_no_decay['args']
        )

        if load_sd:
            optimizer.load_state_dict(config['sd'])

        self.optimizer = optimizer
        
        
    def forward_ar_model(self, z, c):
        input_tokens = z[:, :-1]
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            logits, loss = self.model_ddp(cond_idx=c, idx=input_tokens, targets=z)
        loss = loss.mean()
        return logits, loss
    

    def _iter_step(self, data, is_train):
        x = data.pop('gt').to(self.device, non_blocking=True)
        c = data.pop('label').to(self.device, non_blocking=True)

        # first stage model
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=not self.vae_force_fp32 and self.use_amp):
            with torch.no_grad():
                z = self.vae.encode(x)['bottleneck_rep'] # get the discrete token representation

        # AR modeling
        logits, loss = self.forward_ar_model(z, c)

        # calculate the top1 accuracy and the top5 accuracy
        topk_accuracies = utils.calculate_topk_accuracy(logits, z, topk=(1, 5))

        if is_train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for ema_decay, ema_model in self.ema_model_dict.items():
                self.update_ema(ema_model, decay=ema_decay)

        return_dict = {'loss': loss.item(), **topk_accuracies}
        return return_dict


    def train_step(self, data):
        return self._iter_step(data, is_train=True)


    def evaluate_step(self, data):
        with torch.no_grad():
            return self._iter_step(data, is_train=False)


    @torch.inference_mode()
    def visualize_epoch(self, c_distribution=None, logging=True, use_ema=False, force_fp32=False):
        if use_ema:
            model = self.ema_model_dict[self.ema_decay_list[0]]
            self.log(f'Using EMA model {self.ema_decay_list[0]} for visualization')
        else:
            model = self.model
            self.log('Using current model for visualization')
        model = deepcopy(model)
        if not force_fp32:
            model = model.to(device=self.device, dtype=torch.bfloat16)
        model.eval()

        fp16_enabled = (not force_fp32) and self.use_amp
        self.log(f'Visualizing with fp16: {fp16_enabled}')

        out_dir = os.path.join(self.cfg['env']['save_dir'], 'visualize')
        vid_dir = os.path.join(out_dir, 'vid')
        if self.is_master:
            os.makedirs(vid_dir, exist_ok=True)

        sample_i3d_feats = None
        sample_inception_feats = None

        if dist.is_initialized():
            dist.barrier()
            world_size = dist.get_world_size()
        else:
            world_size = 1

        assert self.sample_batch_size % world_size == 0, "sample_batch_size must be divisible by world_size"
        n = int(self.sample_batch_size // world_size)
        total_samples = int(math.ceil(self.num_samples / self.sample_batch_size) * self.sample_batch_size)

        if self.is_master:
            self.log(f"Total number of images that will be sampled: {total_samples}")

        assert total_samples % world_size == 0, "total_samples must be divisible by world_size"
        samples_needed_this_gpu = int(total_samples // world_size)
        assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"

        iterations = int(samples_needed_this_gpu // n)

        if c_distribution is None:
            assert hasattr(self, 'train_loader'), "train_loader is not defined"
            c_distribution = self.train_loader.dataset.label_count
            if c_distribution is not None:
                c_distribution = torch.tensor(c_distribution, device=self.device, dtype=torch.float32)
                c_distribution /= c_distribution.sum()

        pbar = range(iterations)
        pbar = tqdm(pbar) if self.is_master else pbar

        vis_res = []
        for _ in pbar:
            if c_distribution is not None:
                c = torch.multinomial(c_distribution, n, replacement=True).to(device=self.device)
            else:
                c = -1 * torch.ones(n, device=self.device, dtype=torch.int64)

            sampled_seqs = model.sample(
                c=c,
                cfg_scale=self.cfg_scale, cfg_interval=self.cfg_interval,
                temperature=self.sampling_temperture, 
                top_k=self.sampling_top_k,
                top_p=self.sampling_top_p
            )
            
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                sampled_batch: torch.FloatTensor = self.vae.decode_from_bottleneck(sampled_seqs) # (b, c, t, h, w)

            sampled_batch = sampled_batch.clamp(0., 1.)
            sampled_batch = sampled_batch.to(torch.float32)
            if sampled_batch.shape[2] >= 10: # FVD calc requires at least 10 frames
                sample_i3d_feats = self.fvd_calculator.get_feature_stats_for_batch(sampled_batch, sample_i3d_feats)
            if self.fid_calculator is not None and sampled_batch.shape[2] == 1:
                sample_inception_feats = self.fid_calculator.get_feature_stats_for_batch(sampled_batch.squeeze(), sample_inception_feats)

            if len(vis_res) < self.num_save_wandb and self.is_master:
                vis_res.append(sampled_batch * 255.)

        del model
        
        if dist.is_initialized():
            dist.barrier()

        if sample_i3d_feats is not None:
            assert sample_i3d_feats.num_items == total_samples, f"sample_i3d_feats.num_items={sample_i3d_feats.num_items}, total_samples={total_samples}"
        if sample_inception_feats is not None:
            assert sample_inception_feats.num_items == total_samples, f"sample_inception_feats.num_items={sample_inception_feats.num_items}, total_samples={total_samples}"

        fvd, fid = None, None

        if self.is_master:
            vis_res = torch.cat(vis_res, dim=0)[:self.num_save_wandb]
            vis_res = einops.rearrange(vis_res, 'b c t h w -> b t c h w').cpu()
            vis_res = vis_res.type(torch.uint8)

            assert self.num_save_wandb % 8 == 0, "num_save_wandb must be divisible by 8"
            col = 8
            row = self.num_save_wandb // col

            # concat all samples into a video, 4 rows, 8 columns
            if vis_res.shape[0] < row * col:
                col = 4
                row = vis_res.shape[0] // col

            vis_res = einops.rearrange(vis_res, '(b row col) t c h w -> b t c (row h) (col w)', row=row, col=col)

            if self.enable_wandb:
                if vis_res.shape[1] >= 2: # videos
                    wandb.log({'samples': [wandb.Video(v, fps=4, format="mp4") for v in vis_res]}, step=self.epoch)
                else: # images
                    vis_res = vis_res.squeeze(1)
                    wandb.log({'samples': [wandb.Image(v) for v in vis_res]}, step=self.epoch)

            if sample_i3d_feats is not None and self.fvd_real_stats is not None: 
                self.log('Calculating FVD with loaded real stats')
                fvd = self.fvd_calculator.calculate_fvd(
                    sample_i3d_feats, 
                    self.fvd_real_stats
                )

                if isinstance(fvd, torch.Tensor):
                    fvd = fvd.item()

                if logging:
                    logtext = f' sample_fvd={fvd:.4f}'
                    self.log_temp_scalar(f'test/fvd', fvd)
                    self.log_buffer.append(logtext)
                
                if hasattr(self, 'current_fvd'):
                    self.current_fvd = fvd

            if sample_inception_feats is not None and self.fid_real_stats is not None and self.fid_calculator is not None:
                self.log('Calculating FID with loaded real stats')
                fid = self.fid_calculator.calculate_fid(
                    sample_inception_feats, 
                    self.fid_real_stats
                )

                if isinstance(fid, torch.Tensor):
                    fid = fid.item()

                if logging:
                    logtext = f' sample_fid={fid:.4f}'
                    self.log_temp_scalar(f'test/fid', fid)
                    self.log_buffer.append(logtext)

        if dist.is_initialized():
            dist.barrier()

        return {
            'fvd': fvd,
            'fid': fid
        }


