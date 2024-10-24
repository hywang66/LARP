import glob
import json
import os
import pickle
from hashlib import md5
from pathlib import Path
from typing import Dict, Union

import einops
import numpy as np
import torch
import torch.distributed
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .inception import InceptionV3
from ..fvd.fvd import FeatureStats, trace_sqrt_product

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


def calc_dataset_md5(dataset):
    try:
        md5_val =  md5(json.dumps(dataset.__dict__, sort_keys=True).encode('utf-8')).hexdigest()
    except Exception as e:
        print(f'Failed to calculate md5 for dataset: {e}. Using pickle instead.')
        data_bytes = pickle.dumps(dataset)
        md5_val = md5(data_bytes).hexdigest()
    return md5_val


class ImageDataset(data.Dataset):
    def __init__(self, data_folder: str, resolution: int = 256):
        super().__init__()
        self.resolution = resolution
        folder = data_folder
        self.files = sum([glob.glob(os.path.join(folder, '**', f'*{ext}'), recursive=True)
                     for ext in IMAGE_EXTENSIONS], [])
        
        self.transform = transforms.Compose([
            transforms.Resize(resolution, antialias=True),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = self.transform(img)
        return {'gt': img}


class FIDCalculator:
    def __init__(self, dims=2048, device='cuda', version='stable', capture_all=False):
        self.device = device
        if torch.distributed.is_initialized():
            self.num_gpus = torch.distributed.get_world_size() # use world_size as num_gpus
        else:
            self.num_gpus = torch.cuda.device_count() # use local device count as num_gpus
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device).eval()
        self.capture_all = capture_all

        assert version in ['original', 'stable'], f'Expected version to be `original` or `stable`, got {version}'
        if version == 'original':
            self.calculate_fid = self.calculate_fid_original
        else:
            self.calculate_fid = self.calculate_fid_stable

    @torch.inference_mode()
    def get_feature_stats_for_batch(self, batch, feats=None, num_gpus=None):
        if num_gpus is None:
            num_gpus = self.num_gpus
        if feats is None:
            if self.capture_all:
                feats = FeatureStats(capture_all=True)
            else:
                feats = FeatureStats(capture_mean_cov=True)

        if isinstance(batch, Dict):
            if 'gt' in batch:
                data = batch['gt']
            elif 'video' in batch: # images are also videos, single frame videos
                data = batch['video']
            else:
                raise ValueError('Expected key `gt` or `video` in the batch dict.') 
        else:
            data = batch

        # InceptionV3 assumes that data is in [0, 1]
        assert isinstance(data, torch.Tensor), f'Expected torch.Tensor, got {type(data)}'
        data = data.to(self.device)

        if data.ndim == 5: # video, [B, C, T, H, W]
            data = einops.rearrange(data, 'b c t h w -> (b t) c h w')
        else:
            assert data.ndim == 4, f'Expected 5D (video batch) or 4D (image batch) tensor, got {data.ndim}D tensor'

        features = self.model(data)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))

        features = features.squeeze(3).squeeze(2) # [B, D]
        feats.append_torch(features, num_gpus=num_gpus)
        return feats

    def calculate_fid_original(
        self,
        feats_gen: FeatureStats,
        feats_real: FeatureStats,
        eps=1e-6,
    ):

        mu1, sigma1 = feats_gen.get_mean_cov()
        mu2, sigma2 = feats_real.get_mean_cov()
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


    def calculate_fid_stable(
        self,
        feats_gen: FeatureStats,
        feats_real: FeatureStats,
    ):
        
        mu_gen, cov_gen = feats_gen.get_mean_cov()
        mu_real, cov_real = feats_real.get_mean_cov()

        mu_gen = torch.from_numpy(mu_gen)
        cov_gen = torch.from_numpy(cov_gen)
        mu_real = torch.from_numpy(mu_real)
        cov_real = torch.from_numpy(cov_real)

        mean = torch.sum((mu_gen - mu_real) ** 2)
        sqrt_trace_component = trace_sqrt_product(cov_gen, cov_real)
        trace = torch.trace(cov_gen + cov_real) - 2.0 * sqrt_trace_component
        fid = trace + mean
        return fid
    

    def calculate_stats_for_dataset(self, dataset: ImageDataset, bs=32, num_workers=4):
        assert isinstance(dataset, Dataset), f'Expected a torch Dataset, but got {type(dataset)}'
        feats = FeatureStats(capture_mean_cov=True)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
        for batch in tqdm(loader, desc='Extracting features'):
            feats = self.get_feature_stats_for_batch(batch, feats, num_gpus=1)
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
                stats_pkl = f'fid_stats_{dataset_name}_{dataset_md5}.pkl'
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
                    print(f'Saved stats to {stats_pkl_path}')
            torch.distributed.barrier()
            if rank != 0:
                assert stats_pkl_path.exists()
                feats = FeatureStats.load(stats_pkl_path)

        else:
            feats = self.calculate_stats_for_dataset(dataset, bs, num_workers)
            if cache_stats:
                feats.save(stats_pkl_path)
                print(f'Saved stats to {stats_pkl_path}')

        return feats

    def to_feature_stats(
            self, 
            x: Union[FeatureStats, Dataset, os.PathLike, str],
            bs=32,
            num_workers=4,
            resolution=256,
            cache_stats=False,
        ):
        '''
        Convert input to FeatureStats object.
        x: FeatureStats, Dataset, or path-like object.
        If x is a path-like object, it can be a directory containing images or a .pkl file, which contains a FeatureStats object.
        '''

        if isinstance(x, FeatureStats):
            return x
        elif isinstance(x, Dataset):
            return self.get_feature_stats_for_dataset(x, bs=bs, num_workers=num_workers, cache_stats=cache_stats)
        else:
            try:
                path = Path(x)
            except Exception as e:
                raise ValueError(f'x must be a FeatureStats, Dataset, or a path-like object, but got {type(x)}') from e
            if not path.exists():
                raise FileNotFoundError(f'File not found: {path}')
            if path.is_dir():
                dataset = ImageDataset(data_folder=path, resolution=resolution)
                return self.get_feature_stats_for_dataset(dataset, bs=bs, num_workers=num_workers, cache_stats=cache_stats)
            else:
                try:
                    feats = FeatureStats.load(path)
                except Exception as e:
                    raise ValueError(f'Failed to load FeatureStats from {path}: {e}') from e
                return feats

    def calculate_fid_smart(
        self,
        gen: Union[FeatureStats, Dataset, os.PathLike, str],
        real: Union[FeatureStats, Dataset, os.PathLike, str],
        bs=32,
        num_workers=4,
        resolution=256,
        cache_stats=False,
    ):
        gen_feats = self.to_feature_stats(gen, bs=bs, num_workers=num_workers, resolution=resolution, cache_stats=cache_stats)
        real_feats = self.to_feature_stats(real, bs=bs, num_workers=num_workers, resolution=resolution, cache_stats=cache_stats)
        fid = self.calculate_fid(gen_feats, real_feats)
        return fid
