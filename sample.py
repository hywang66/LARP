import argparse
import math
import os
import pdb
import random
import time
from concurrent.futures import ThreadPoolExecutor

import einops
import filelock
import imageio
import pandas as pd
import torch
import torchvision
from einops import rearrange
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import datasets
import utils
from eval import calc_fvd_from_multiple_feature_stats
from models.larp_ar import LARP_AR
from models.larp_tokenizer import LARPTokenizer
from utils import FVDCalculator


def get_calculating_flag_file_path(
    ar_model_id,
    num_samples=None,
    cfg_scale=None,
):
    project_base_dir = os.path.dirname(__file__)
    tmp_dir = os.path.join(project_base_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    flag_name = 'flag_sampling_' + ar_model_id.split('/')[-1]
    if num_samples is not None:
        flag_name += f'_n{num_samples}'
    if cfg_scale is not None:
        flag_name += f'_cfg{cfg_scale}'
    return os.path.join(tmp_dir, f"{flag_name}.txt")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description='Sample videos')
    
    parser.add_argument('--ar_model', type=str, required=True, help='AR model ID')
    parser.add_argument('--tokenizer', type=str, required=True, help='Tokenizer model ID')
    
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save the samples')
    parser.add_argument('--num_samples', '--ns', type=int, default=1024, help='Number of samples to generate')
    parser.add_argument('--starting_index', type=int, default=0, help='Starting index for the samples')
    parser.add_argument('--sample_batch_size', type=int, default=32, help='Batch size for sampling')
    parser.add_argument('--cfg_scale', '--cs', type=float, default=1.0, help='Scale for the classifier free guidance')
    parser.add_argument('--cfg_interval', type=int, default=-1, help='Interval for the classifier free guidance')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k for sampling')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p for sampling')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type for the samples')
    parser.add_argument('--num_samples_total', type=int, default=None, help='Total number of samples to generate in parallel')
    parser.add_argument('--dataset_csv', type=str, default='ucf101_train.csv', help='Path to the dataset csv')
    parser.add_argument('--dataset_split_seed', type=int, default=42, help='Seed for splitting the dataset')
    parser.add_argument('--num_cond_frames', type=int, default=None, help='Number of frames to condition on for frame prediction')
    
    parser.add_argument('--frame_prediction', '--fp', action='store_true', help='Predict frames instead of sampling videos')
    parser.add_argument('--stats_only', action='store_true', help='Only compute stats, do not save samples')
    parser.add_argument('--replace', action='store_true', help='Replace existing samples and results')

    args = parser.parse_args(input_args)
    return args


def save_video(video, path):
    imageio.mimwrite(path, video.numpy(), fps=25)

@torch.inference_mode()
def sample_videos(
    larp_ar_model: LARP_AR,
    larp_tokenizer: LARPTokenizer,
    output_dir: str,
    dataset_csv: str,
    num_samples=10000,
    starting_index=0,
    sample_batch_size=16,
    cfg_scale=1.0,
    cfg_interval=-1,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    dtype=torch.bfloat16,
    dataset_split_seed=42,
    num_samples_total=None,
    stats_only=False,
):
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    print(f'Using dataset csv: {dataset_csv}')
    print(f'Saving samples to {output_dir}...')

    if not isinstance(dtype, torch.dtype):
        assert isinstance(dtype, str), f"Invalid dtype: {dtype}"
        dtype = getattr(torch, dtype)

    larp_ar_model = larp_ar_model.to(dtype)
    larp_ar_model.eval()

    larp_tokenizer.to(torch.float16)  # force fp16 for LARP tokenizer
    larp_tokenizer.eval()

    assert larp_ar_model.device == larp_tokenizer.device, "Models must be on the same device"
    device = larp_ar_model.device

    # Prepare output directories
    os.makedirs(output_dir, exist_ok=True)
    if not stats_only:
        output_sample_dir = os.path.join(output_dir, 'sampled_videos')
        os.makedirs(output_sample_dir, exist_ok=True)
        executor = ThreadPoolExecutor(max_workers=4)

    # Load dataset
    dataset_cfg = {
        'name': 'video_dataset',
        'args': {
            'root_path': 'data/metadata',
            'frame_num': larp_tokenizer.frame_num,
            'cls_vid_num': '-1_-1',
            'crop_size': larp_tokenizer.input_size,
            'csv_file': dataset_csv,
            'frame_rate': 'native',
            'use_all_frames': True,
            'pre_load': False,
        },
    }
    sample_dataset = datasets.make(dataset_cfg)
    if num_samples_total is None:
        num_samples_total = num_samples
    dataset_idx_seq = random.Random(dataset_split_seed).sample(range(len(sample_dataset)), num_samples_total)
    sample_dataset = Subset(sample_dataset, dataset_idx_seq[starting_index:starting_index + num_samples])

    sample_loader = DataLoader(
        sample_dataset,
        batch_size=sample_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )
    pbar = tqdm(sample_loader)

    sample_i3d_feats = None
    orig_i3d_feats = None
    fvd_calculator = FVDCalculator(device=device)

    # Paths for saving FVD stats
    gt_fvd_stats_path = os.path.join(output_dir, f'gt_fvd_stats_{starting_index}_{starting_index + num_samples}.pkl')
    generated_fvd_stats_path = os.path.join(output_dir, f'generated_fvd_stats_{starting_index}_{starting_index + num_samples}.pkl')


    loss_list = []
    start_time = time.time()

    for i, data in enumerate(pbar):
        c = data['label'].to(device)
        orig_video = data['gt'].clamp(0., 1.).to(device)

        # Sample sequences
        sampled_seqs = larp_ar_model.sample(
            c=c,
            cfg_scale=cfg_scale, cfg_interval=cfg_interval,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        # Compute loss
        if hasattr(larp_ar_model, 'reset_caches'):
            larp_ar_model.reset_caches()
        _, loss = larp_ar_model(
            cond_idx=c,
            idx=sampled_seqs[:, :-1],
            targets=sampled_seqs.to(dtype=torch.int64).contiguous()
        )
        loss_list.append(loss)

        # Decode sequences
        sampled_batch = larp_tokenizer.decode_from_bottleneck(sampled_seqs)  # (b, c, t, h, w)
        samples = sampled_batch.float().clamp(0., 1.)

        if (i + 1) * sample_batch_size > num_samples:
            samples = samples[:num_samples - i * sample_batch_size]
            orig_video = orig_video[:num_samples - i * sample_batch_size]

        # Compute FVD stats
        sample_i3d_feats = fvd_calculator.get_feature_stats_for_batch(samples, sample_i3d_feats)
        orig_i3d_feats = fvd_calculator.get_feature_stats_for_batch(orig_video, orig_i3d_feats)

        if not stats_only:
            samples_cpu = samples.cpu()
            videos = einops.rearrange(samples_cpu, 'b c t h w -> b t h w c') * 255.
            videos = videos.type(torch.uint8)
            for j, video in enumerate(videos):
                if i * sample_batch_size + j >= num_samples:
                    break
                video_path = os.path.join(output_sample_dir, f'{starting_index + i * sample_batch_size + j}.mp4')
                executor.submit(save_video, video, video_path)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f}s, sampled {num_samples} videos")
    print(f"Samples per second: {num_samples / (end_time - start_time):.2f}")

    orig_i3d_feats.save(gt_fvd_stats_path)
    sample_i3d_feats.save(generated_fvd_stats_path)
    print(f'Saved gt FVD stats to {gt_fvd_stats_path}')
    print(f'Saved generated FVD stats to {generated_fvd_stats_path}')

    if not stats_only:
        executor.shutdown(wait=True)
        print(f"Saved samples to {output_dir}")

    loss_mean = torch.mean(torch.stack(loss_list)).item()
    return loss_mean




@torch.inference_mode()
def predict_frames(
    larp_ar_model: LARP_AR,
    larp_tokenizer: LARPTokenizer,
    output_dir: str,
    dataset_csv: str,
    num_cond_frames=5,
    num_samples=50000,
    starting_index=0,
    sample_batch_size=16,
    cfg_scale=1.0,
    cfg_interval=-1,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    dtype=torch.bfloat16,
    dataset_split_seed=42,
    num_samples_total=None,
    resolution=64,
    stats_only=False,
):
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    assert cfg_scale == 1.0, "frame prediction requires cfg_scale=1.0 (no classifier-free guidance)"
    assert num_cond_frames is not None, "num_cond_frames must be specified for frame prediction"
    assert num_cond_frames > 0, "num_cond_frames must be positive"

    print(f'Using dataset csv: {dataset_csv}')
    print(f'Saving samples to {output_dir}...')

    if not isinstance(dtype, torch.dtype):
        assert isinstance(dtype, str), f"Invalid dtype: {dtype}"
        dtype = getattr(torch, dtype)


    larp_ar_model = larp_ar_model.to(dtype)
    larp_ar_model.eval()

    larp_tokenizer.to(torch.float32) # force fp32 for LARP tokenizer
    larp_tokenizer.eval()

    assert larp_ar_model.device == larp_tokenizer.device, "Models must be on the same device"
    device = larp_ar_model.device

    # Prepare output directories
    os.makedirs(output_dir, exist_ok=True)
    if not stats_only:
        output_sample_dir = os.path.join(output_dir, 'sampled_videos')
        os.makedirs(output_sample_dir, exist_ok=True)
        output_gt_dir = os.path.join(output_dir, 'gt')
        os.makedirs(output_gt_dir, exist_ok=True)
        executor = ThreadPoolExecutor(max_workers=4)


    # make the dataset for conditioning
    dataset_cfg = {
        'name': 'video_dataset',
        'args': {
            'root_path': 'data/metadata',
            'frame_num': larp_tokenizer.frame_num,
            'cls_vid_num': '-1_-1',
            'crop_size': larp_tokenizer.input_size,
            'csv_file': dataset_csv,
            'frame_rate': 'native',
            'use_all_frames': True,
            'pre_load': False,
        },
    }

    sample_dataset = datasets.make(dataset_cfg)
    if num_samples_total is None:
        num_samples_total = num_samples
    dataset_idx_seq = random.Random(dataset_split_seed).sample(range(len(sample_dataset)), num_samples_total)
    sample_dataset = Subset(sample_dataset, dataset_idx_seq[starting_index:starting_index + num_samples])

    sample_loader = DataLoader(
        sample_dataset,
        batch_size=sample_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )
    pbar = tqdm(sample_loader)

    sample_i3d_feats = None
    orig_i3d_feats = None
    fvd_calculator = FVDCalculator(device=device)

    # Paths for saving FVD stats
    gt_fvd_stats_path = os.path.join(output_dir, f'gt_fvd_stats_{starting_index}_{starting_index + num_samples}.pkl')
    generated_fvd_stats_path = os.path.join(output_dir, f'generated_fvd_stats_{starting_index}_{starting_index + num_samples}.pkl')

    resize_video_transform = torchvision.transforms.Resize((resolution, resolution))
    def resize_video(video):
        # video: (b, c, t, h, w)
        t = video.shape[2]
        video = rearrange(video, 'b c t h w -> (b t) c h w')
        video = resize_video_transform(video)
        video = rearrange(video, '(b t) c h w -> b c t h w', t=t)
        return video

    start_time = time.time()

    for i, data in enumerate(pbar):
        x = data['gt'].to(device=larp_tokenizer.device, dtype=larp_tokenizer.dtype, non_blocking=True)
        x_cond = utils.repeat_to_m_frames(x[:, :, :num_cond_frames], m=larp_tokenizer.frame_num)
        c = larp_tokenizer.encode(x_cond)['bottleneck_rep'] # get the discrete token representation for conditioning
        sep_token = torch.full((c.shape[0], 1), larp_tokenizer.codebook_size, device=c.device, dtype=c.dtype)
        c = torch.cat([c, sep_token], dim=1)

        sampled_seqs = larp_ar_model.sample(
            c=c,
            cfg_scale=cfg_scale, cfg_interval=cfg_interval,
            temperature=temperature, 
            top_k=top_k,
            top_p=top_p
        )

        if hasattr(larp_ar_model, 'reset_caches'):
            larp_ar_model.reset_caches()

        sampled_batch = larp_tokenizer.decode_from_bottleneck(sampled_seqs).float() # (b, c, t, h, w)  
        sampled_batch = resize_video(sampled_batch.clamp(0., 1.)).contiguous() # 64 x 64 spatial resolution
        orig_video = resize_video(data['gt'].clamp(0., 1.)).contiguous() # 64 x 64 spatial resolution

        # Compute FVD stats
        sample_i3d_feats = fvd_calculator.get_feature_stats_for_batch(sampled_batch, sample_i3d_feats) 
        orig_i3d_feats = fvd_calculator.get_feature_stats_for_batch(orig_video, orig_i3d_feats) 
        
        if not stats_only:
            samples = sampled_batch.cpu() # (b, c, t, hs, ws)
            orig_video = orig_video.cpu()
            videos = einops.rearrange(samples, 'b c t h w -> b t h w c') * 255.
            videos = videos.type(torch.uint8)
            orig_videos = einops.rearrange(orig_video, 'b c t h w -> b t h w c') * 255.
            orig_videos = orig_videos.type(torch.uint8)

            for j, video in enumerate(videos):
                if i * sample_batch_size + j >= num_samples:
                    break
                video_path = os.path.join(output_sample_dir, f'{starting_index + i * sample_batch_size + j}.mp4')
                executor.submit(save_video, video, video_path)
                gt_video_path = os.path.join(output_gt_dir, f'{starting_index + i * sample_batch_size + j}.mp4')
                executor.submit(save_video, orig_videos[j], gt_video_path)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f}s, sampled {num_samples} videos")
    print(f"Samples per second: {num_samples / (end_time - start_time):.2f}")

    orig_i3d_feats.save(gt_fvd_stats_path)
    sample_i3d_feats.save(generated_fvd_stats_path)
    print(f'Saved gt FVD stats to {gt_fvd_stats_path}')
    print(f'Saved generated FVD stats to {generated_fvd_stats_path}')

    if not stats_only:
        executor.shutdown(wait=True)
        print(f"Saved samples to {output_dir}")
    return 




def main(args):
    rgsd = parse_args([
        '--ar_model', '/dev/null',
        '--tokenizer', '/dev/null'
    ])
    for attr, value in vars(rgsd).items():
        if not hasattr(args, attr):
            setattr(args, attr, value)

    if args.num_samples_total is None:
        args.num_samples_total = args.num_samples
    num_jobs = math.ceil(args.num_samples_total / args.num_samples)

    if os.path.exists(args.ar_model):
        # load model from local checkpoint
        larp_ar_model = LARP_AR.from_checkpoint(args.ar_model).cuda()
    else:
        # load model from huggingface hub
        larp_ar_model = LARP_AR.from_pretrained(args.ar_model).cuda()
    if os.path.exists(args.tokenizer):
        # load model from local checkpoint
        larp_tokenizer = LARPTokenizer.from_checkpoint(args.tokenizer).cuda()
    else:
        # load model from huggingface hub
        larp_tokenizer = LARPTokenizer.from_pretrained(args.tokenizer).cuda()

    if args.frame_prediction:
        loss_mean = predict_frames(
            larp_ar_model=larp_ar_model,
            larp_tokenizer=larp_tokenizer,
            output_dir=args.output_dir,
            dataset_csv=args.dataset_csv,
            num_cond_frames=args.num_cond_frames,
            num_samples=args.num_samples_total,
            starting_index=args.starting_index,
            sample_batch_size=args.sample_batch_size,
            cfg_scale=args.cfg_scale,
            cfg_interval=args.cfg_interval,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            dtype=args.dtype,
            dataset_split_seed=args.dataset_split_seed,
            num_samples_total=args.num_samples_total,
            resolution=64,
            stats_only=args.stats_only,
        )
    else:
        loss_mean = sample_videos(
            larp_ar_model=larp_ar_model,
            larp_tokenizer=larp_tokenizer,
            output_dir=args.output_dir,
            dataset_csv=args.dataset_csv,
            num_samples=args.num_samples,
            starting_index=args.starting_index,
            sample_batch_size=args.sample_batch_size,
            cfg_scale=args.cfg_scale,
            cfg_interval=args.cfg_interval,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            dtype=args.dtype,
            dataset_split_seed=args.dataset_split_seed,
            num_samples_total=args.num_samples_total,
            stats_only=args.stats_only,
        )


    calculating_flag_file_path = get_calculating_flag_file_path(
        ar_model_id=args.ar_model,
        num_samples=args.num_samples_total,
        cfg_scale=args.cfg_scale,
    )

    num_pkl_finally = num_jobs * 2 # 2 for generated and gt
    if args.replace:
        if os.path.exists(calculating_flag_file_path):
            os.remove(calculating_flag_file_path)
        if os.path.exists(calculating_flag_file_path + ".lock"):
            os.remove(calculating_flag_file_path + ".lock")

    with filelock.FileLock(calculating_flag_file_path + ".lock"):
        num_pkl = len([f for f in os.listdir(args.output_dir) if f.lower().endswith('.pkl')])
        if num_pkl == num_pkl_finally and not os.path.exists(calculating_flag_file_path): # no other process has already calculated FVD and all samples are generated
            with open(calculating_flag_file_path, "w") as f:
                f.write("started")
        else:
            print('INFO: FVD calculation skipped because either not all samples have been generated, or another process is already calculating it.')
            return loss_mean 
        
    if num_jobs > 1:
        time.sleep(10) # wait for a few seconds to be safe in multi-job settings

    args.feature_stats_dir = args.output_dir
    fvd = calc_fvd_from_multiple_feature_stats.main(args)
    fvd_str = f"{fvd:.4f}"

    os.remove(calculating_flag_file_path)
    os.remove(calculating_flag_file_path + ".lock")

    project_base_dir = os.path.dirname(__file__)
    report_path = os.path.join(project_base_dir, "fvd_report.csv")
    

    df_this = pd.DataFrame({
        "ar_model": [args.ar_model],
        "tokenizer": [args.tokenizer],
        "num_samples": [args.num_samples_total],
        "cfg_scale": [args.cfg_scale],
        "temperature": [args.temperature],
        "top_k": [args.top_k],
        "top_p": [args.top_p],
        "dtype": [args.dtype],
        "NLL_sampled": [loss_mean],
        "fvd": [fvd_str],
    })

    with filelock.FileLock(report_path + ".lock"):
        if os.path.exists(report_path):
            orig_df = pd.read_csv(report_path)
            df_local = pd.concat([orig_df, df_this], ignore_index=True)
            df_local = df_local.sort_values(by=['num_samples', 'fvd'], key=lambda x: x.astype(float), ascending=[False, True])
        else:
            df_local = df_this
        df_local.to_csv(report_path, index=False)

    print(f"Saved FVD report to {report_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)


