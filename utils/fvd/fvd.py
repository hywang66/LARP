import glob
import json
import math
import os
import pickle
import random
import warnings
from hashlib import md5
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils.data as data
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.video_utils import VideoClips
from tqdm import tqdm


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())

# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def calc_dataset_md5(dataset):
    try:
        md5_val =  md5(json.dumps(dataset.__dict__, sort_keys=True).encode('utf-8')).hexdigest()
    except Exception as e:
        print(f'Failed to calculate md5 for dataset: {e}. Using pickle instead.')
        data_bytes = pickle.dumps(dataset)
        md5_val = md5(data_bytes).hexdigest()
    return md5_val

class FeatureStats:

    def __init__(
        self,
        capture_all=False,
        capture_mean_cov=False,
        max_items=None,
        only_stats_mode=False,
        loaded_mean=None,
        loaded_cov=None,
    ):
        self.only_stats_mode = only_stats_mode
        if only_stats_mode:
            # load pre-computed mean and cov
            assert loaded_mean is not None and loaded_cov is not None, 'loaded_mean and loaded_cov must be provided in only_stats_mode'
            self.loaded_mean = loaded_mean
            self.loaded_cov = loaded_cov
        else:
            assert loaded_mean is None and loaded_cov is None, 'loaded_mean and loaded_cov must be None if only_stats_mode is False'
            self.loaded_mean = self.loaded_cov = None
            self.capture_all = capture_all
            self.capture_mean_cov = capture_mean_cov
            self.max_items = max_items
            self.num_items = 0
            self.num_features = None
            self.all_features = None
            self.raw_mean = None
            self.raw_cov = None

    def set_num_features(self, num_features):
        if self.only_stats_mode:
            raise ValueError('Cannot set num_features in only_stats_mode')

        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        if self.only_stats_mode:
            return True
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        if self.only_stats_mode:
            raise ValueError('Cannot append in only_stats_mode')

        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1):
        if self.only_stats_mode:
            raise ValueError('Cannot append in only_stats_mode')

        assert isinstance(x, torch.Tensor) and x.ndim == 2
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.float().cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        if self.only_stats_mode:
            return self.loaded_mean, self.loaded_cov
        else:
            if self.capture_mean_cov:
                mean = self.raw_mean / self.num_items
                cov = self.raw_cov / self.num_items
                cov = cov - np.outer(mean, mean)
                
            elif self.capture_all:
                features = self.get_all()
                mean = np.mean(features, axis=0)
                cov = np.cov(features, rowvar=False)

            else:
                raise ValueError('No stats captured')
            
            return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(stats_path):
        stats_path = Path(stats_path)
        if stats_path.suffix == '.pkl': # pickle file, load as FeatureStats
            with open(stats_path, 'rb') as f:
                s = edict(pickle.load(f))
            obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
            obj.__dict__.update(s)
        elif stats_path.suffix == '.npz': # npz file, ADM's precomputed mean and cov
            data = np.load(stats_path)
            obj = FeatureStats(only_stats_mode=True, loaded_mean=data['mu'], loaded_cov=data['sigma'])
        else:
            raise ValueError(f'Unknown file extension: {stats_path}')
        return obj
    

    def __add__(self, other):
        if not isinstance(other, FeatureStats):
            return NotImplemented

        if self.only_stats_mode or other.only_stats_mode:
            raise ValueError('Cannot add FeatureStats with only_stats_mode=True')

        if self.num_features != other.num_features:
            raise ValueError('Cannot add FeatureStats with different num_features')

        # Check compatibility of capture_all and capture_mean_cov
        if self.capture_all != other.capture_all:
            raise ValueError('Cannot add FeatureStats with different capture_all settings')

        if self.capture_mean_cov != other.capture_mean_cov:
            raise ValueError('Cannot add FeatureStats with different capture_mean_cov settings')

        # Create a new FeatureStats instance
        result = FeatureStats(
            capture_all=self.capture_all,
            capture_mean_cov=self.capture_mean_cov,
            max_items=None,  # No limit for the merged result
            only_stats_mode=False
        )
        result.num_features = self.num_features
        result.num_items = self.num_items + other.num_items

        # Combine all_features if capture_all is True
        if self.capture_all:
            result.all_features = self.all_features + other.all_features
        else:
            result.all_features = None

        # Combine raw_mean and raw_cov if capture_mean_cov is True
        if self.capture_mean_cov:
            result.raw_mean = self.raw_mean + other.raw_mean
            result.raw_cov = self.raw_cov + other.raw_cov
        else:
            result.raw_mean = None
            result.raw_cov = None

        return result


def preprocess(video, resolution, sequence_length=None, in_channels=3, sample_every_n_frames=1):
    # video: THWC, {0, ..., 255}
    assert in_channels == 3
    video = video.permute(0, 3, 1, 2).float() / 255.  # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # skip frames
    if sample_every_n_frames > 1:
        video = video[::sample_every_n_frames]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False, antialias=True)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous()  # CTHW

    return {'video': video}


VID_EXTENSIONS = ['.avi', '.mp4', '.webm', '.mov', '.mkv', '.m4v']


class VideoDataset(data.Dataset):
    """ 
    Generic dataset for videos files stored in folders.
    Videos of the same class are expected to be stored in a single folder. Multiple folders can exist in the provided directory.
    The class depends on `torchvision.datasets.video_utils.VideoClips` to load the videos.
    Returns BCTHW videos in the range [0, 1].

    Args:
        data_folder: Path to the folder with corresponding videos stored.
        sequence_length: Length of extracted video sequences.
        resolution: Resolution of the returned videos.
        sample_every_n_frames: Sample every n frames from the video.
    """

    def __init__(self, data_folder: str, sequence_length: int = 16, resolution: int = 128, sample_every_n_frames: int = 1):
        super().__init__()
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.sample_every_n_frames = sample_every_n_frames

        folder = data_folder
        files = sum([glob.glob(os.path.join(folder, '**', f'*{ext}'), recursive=True)
                     for ext in VID_EXTENSIONS], [])
    
        warnings.filterwarnings('ignore')
        cache_file = os.path.join(folder, f"metadata_{sequence_length}.pkl")
        if not os.path.exists(cache_file):
            clips = VideoClips(files, sequence_length, num_workers=4)
            try:
                pickle.dump(clips.metadata, open(cache_file, 'wb'))
            except:
                print(f"Failed to save metadata to {cache_file}")
        else:
            metadata = pickle.load(open(cache_file, 'rb'))
            clips = VideoClips(files, sequence_length,
                               _precomputed_metadata=metadata)

        self._clips = clips
        # instead of uniformly sampling from all possible clips, we sample uniformly from all possible videos
        self._clips.get_clip_location = self.get_random_clip_from_video
        
    def get_random_clip_from_video(self, idx: int) -> tuple:
        '''
        Sample a random clip starting index from the video.

        Args:
            idx: Index of the video.
        '''
        # Note that some videos may not contain enough frames, we skip those videos here.
        while self._clips.clips[idx].shape[0] <= 0:
            idx += 1
        n_clip = self._clips.clips[idx].shape[0]
        clip_id = random.randint(0, n_clip - 1)
        return idx, clip_id

    def __len__(self):
        return self._clips.num_videos()

    def __getitem__(self, idx):
        resolution = self.resolution
        while True:
            try:
                video, _, _, idx = self._clips.get_clip(idx)
            except Exception as e:
                print(idx, e)
                idx = (idx + 1) % self._clips.num_clips()
                continue
            break

        return dict(**preprocess(video, resolution, sample_every_n_frames=self.sample_every_n_frames))


class FVDCalculator:
    def __init__(self, i3d_path=None, device='cuda'):
        if i3d_path is None:
            # https://github.com/universome/fvd-comparison/blob/master/compare_models.py#L34
            # Downloaded from https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1
            i3d_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i3d_torchscript.pt')
        assert os.path.exists(i3d_path), f'Could not find i3d model at {i3d_path}'
        self.device = device
        self.i3d = torch.jit.load(i3d_path).eval().to(device)
        if torch.distributed.is_initialized():
            self.num_gpus = torch.distributed.get_world_size() # use world_size as num_gpus
        else:
            self.num_gpus = torch.cuda.device_count() # use local device count as num_gpus

    @torch.inference_mode()
    def get_feature_stats_for_batch(self, batch, feats=None, num_gpus=None):
        if num_gpus is None:
            num_gpus = self.num_gpus
        if feats is None:
            feats = FeatureStats(capture_mean_cov=True)

        if isinstance(batch, Dict):
            if 'gt' in batch:
                data = batch['gt']
            elif 'video' in batch:
                data = batch['video']
            else:
                raise ValueError('Expected key `gt` or `video` in the batch dict.')
        else:
            data = batch
            
        # Note: the used i3d torchscript model expects input in [-1, 1] and size=224x224.
        # if setting resize=True, the input will be resized to 224x224
        # if setting rescale=True, model expects input in [0, 255] and the model will rescale 
        # the input to [-1, 1] internally.
            
        # Here we assume data is in [0, 1] and BCTHW
        # so wee need to rescale data to [-1, 1] without setting rescale        
        assert isinstance(data, torch.Tensor) and data.ndim == 5 # BCTHW
        data = (data - 0.5) * 2
        features = self.i3d(data.to('cuda'), resize=True, return_features=True)
        feats.append_torch(features, num_gpus=num_gpus)
        return feats
    
    def get_feature_stats_for_dataset(
            self, 
            dataset: Dataset, 
            bs=32, 
            cache_stats=True,
            num_workers=4,
            stats_pkl_path=None
        ): # always using a single gpu
        assert isinstance(dataset, Dataset), f'Expected a torch Dataset, but got {type(dataset)}'
        
        if hasattr(dataset, 'csv_file'):
            dataset_name = Path(dataset.csv_file).stem 
        else:
            dataset_name = 'unknown'

        if cache_stats:
            if stats_pkl_path is None:
                dataset_md5 = calc_dataset_md5(dataset)
                stats_pkl = f'fvd_stats_{dataset_name}_{dataset_md5}.pkl'
                stats_cache_path = Path(__file__).resolve().parent / 'stats_cache'
                stats_cache_path.mkdir(exist_ok=True)
                stats_pkl_path = stats_cache_path / stats_pkl

            if stats_pkl_path.exists():
                return FeatureStats.load(stats_pkl_path)
        
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            if rank == 0:
                feats = self.calculate_stats_for_dataset(dataset, bs, num_workers)
                if cache_stats:
                    feats.save(stats_pkl_path)
            torch.distributed.barrier()
            if rank != 0:
                assert stats_pkl_path.exists()
                feats = FeatureStats.load(stats_pkl_path)

        else:
            feats = self.calculate_stats_for_dataset(dataset, bs, num_workers)
            if cache_stats:
                feats.save(stats_pkl_path)
        
        return feats
    
    def calculate_stats_for_dataset(self, dataset, bs=32, num_workers=4):
        feats = FeatureStats(capture_mean_cov=True)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
        for batch in tqdm(loader, desc='Extracting features'):
            feats = self.get_feature_stats_for_batch(batch, feats, num_gpus=1)
        return feats

    def calculate_fvd(self, feats_gen, feats_real):
        mu_gen, cov_gen = feats_gen.get_mean_cov()
        mu_real, cov_real = feats_real.get_mean_cov()

        # updated for better numerical stability
        mu_gen = torch.from_numpy(mu_gen)
        cov_gen = torch.from_numpy(cov_gen)
        mu_real = torch.from_numpy(mu_real)
        cov_real = torch.from_numpy(cov_real)

        mean = torch.sum((mu_gen - mu_real) ** 2)
        sqrt_trace_component = trace_sqrt_product(cov_gen, cov_real)
        trace = torch.trace(cov_gen + cov_real) - 2.0 * sqrt_trace_component
        fvd = trace + mean
            
        return fvd

    def calculate_fvd_with_dataset(self, feats_gen, dataset_real, bs=32, cache_stats=True):
        feats_real = self.get_feature_stats_for_dataset(dataset_real, bs, cache_stats)
        return self.calculate_fvd(feats_gen, feats_real)
    
    def calculate_fvd_with_video_folder(
        self, 
        feats_real, 
        video_folder, 
        bs=32,
        num_workers=4,
        sequence_length=16, 
        resolution=128,
        cache_stats=False
    ):
        dataset_gen = VideoDataset(
            data_folder=video_folder,
            sequence_length=sequence_length,
            resolution=resolution,
            sample_every_n_frames=1
        )
        feats_gen = self.get_feature_stats_for_dataset(dataset_gen, bs, cache_stats, num_workers=num_workers)
        return self.calculate_fvd(feats_gen, feats_real)
