import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import decord
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import (CenterCrop, RandAugment,
                                    RandomHorizontalFlip, RandomResizedCrop,
                                    Resize)
from tqdm import tqdm

from datasets import register

decord.bridge.set_bridge('torch')


def func_none():
    return None


def read_video_with_retry(uri, retries=5, delay=1):
    for i in range(retries):
        try:
            vr = decord.VideoReader(uri)
            return vr
        except Exception as e:
            print(f"Error reading {uri}, retrying ({i+1}/{retries})...")
            time.sleep(delay)
    raise RuntimeError(f"Failed to read {uri} after {retries} retries")


def VideoTransform(crop_size=128, scale=1.00, ratio=1.00, eval_tfm=False, 
    rand_flip='no'):
    if eval_tfm:
        transform = transforms.Compose([Resize(size=crop_size, antialias=True), CenterCrop(crop_size)])
    else:
        if scale == 1.0 and ratio == 1.0:
            tfm_list = [Resize(size=crop_size, antialias=True), CenterCrop(crop_size)]
        else:
            tfm_list = [Resize(size=int(crop_size/scale), antialias=True), 
                RandomResizedCrop(crop_size, (1./scale**2, 1), (1./ratio, ratio), antialias=True)]
        if rand_flip != 'no':
            tfm_list.append(RandomHorizontalFlip())
        transform = transforms.Compose(tfm_list)

    return transform    

@register('video_dataset')
class VideoDataset(Dataset):

    def __init__(
        self,
        root_path,
        frame_num,
        cls_vid_num,
        crop_size,
        rand_flip='no',
        split='train',
        csv_file='',
        scale=1.0,
        aspect_ratio=1.0,
        rand_augment='no',
        frame_rate='native', # 'uniform' or 'native'
        test_group=0,
        use_all_frames=False,
        pre_load=False
    ):
        self.csv_file = csv_file
        self.frame_num = frame_num
        self.crop_size = crop_size
        assert frame_rate in ['uniform', 'native']
        self.frame_rate = frame_rate
        self.test_group = test_group
        self.use_all_frames = use_all_frames
        self.num_classes = None
        self.label2action = None
        self.action2label = None
        self.vid2label = defaultdict(func_none)

        if csv_file.lower().startswith('null'): # fake dataset to test 
            if csv_file.lower().startswith('null128'):
                num = 128
            else:
                num = 32*7000
            self.fake = True
            self.vid_list = ['' for _ in range(num)]
            self.augment = None
            self.pre_load = False
            self.split = split
            self.rand_flip = rand_flip
            self.crop_size = crop_size
            self.scale = scale
            self.aspect_ratio = aspect_ratio
            self.frame_num = frame_num
            self.index_map_cache_dir = os.path.join(root_path, 'index_map_cache')
            self.idx2label = {i: i % 101 for i in range(num)}
            all_labels = list(self.idx2label.values())
            self.num_classes = 101
            self.label_count = [all_labels.count(label) for label in range(self.num_classes)]
            if self.split == 'train':
                self.cur_tfm = VideoTransform(crop_size=self.crop_size, scale=self.scale, 
                    ratio=self.aspect_ratio, eval_tfm=False)
            elif self.split == 'test':
                self.cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
            else:
                raise NotImplementedError(f'Unknown split: {self.split}')
            
            return

        
        self.fake = False

        if '+' in csv_file:
            self.multiple_datasets = True
            csv_files = csv_file.split('+')
            if cls_vid_num == '-1_-1':
                cls_vid_num = '+'.join(['-1_-1'] * len(csv_files))
            assert '+' in cls_vid_num, 'cls_vid_num should be separated by +'
            cls_vid_nums = cls_vid_num.split('+')
            assert len(csv_files) == len(cls_vid_nums), 'Number of csv_files should be the same as cls_vid_nums'
        else:
            self.multiple_datasets = False
            csv_files = [csv_file]
            cls_vid_nums = [cls_vid_num]
        self.vid_list = []

        self.index_map_cache_dir = os.path.join(root_path, 'index_map_cache')
        os.makedirs(self.index_map_cache_dir, exist_ok=True)

        for csv_file, cls_vid_num in zip(csv_files, cls_vid_nums):
            if csv_file != '':
                if not os.path.isabs(csv_file):
                    csv_file = os.path.join(root_path, csv_file)
                cls_num, vid_num = [int(x) for x in cls_vid_num.split('_')]
                if csv_file.endswith('.csv'):
                    self.process_csv_data(csv_file, cls_num, vid_num)
                elif csv_file.endswith('.js'):
                    with open(csv_file, 'r') as f:
                        vid_dict = json.load(f)
                    sorted_keys=sorted(vid_dict, key=lambda k: len(vid_dict[k]), reverse=True)
                    vid_list = [vid_dict[cls][:vid_num] for cls in sorted_keys[:cls_num]]
                    self.vid_list += sum(vid_list, [])
            else:
                vid_list = []
                cls_num, vid_num = [int(x) for x in cls_vid_num.split('_')]
                root_path = os.path.join(root_path, split)
                for cur_cls in sorted(os.listdir(root_path)[:cls_num]):
                    cur_dir = os.path.join(root_path, cur_cls)
                    for cur_vid in sorted(os.listdir(cur_dir))[:vid_num]:
                        vid_list.append(os.path.join(cur_dir, cur_vid))
                self.vid_list += vid_list

        self.vid_list = sorted(self.vid_list)
        self.split, self.frame_num, self.rand_flip = split, frame_num, rand_flip
        self.crop_size, self.scale, self.aspect_ratio = crop_size, scale, aspect_ratio
        if rand_augment in ['no', '']:
            self.augment = None
        else:
            num_ops, magnitude, num_magnitude_bins = [int(x) for x in rand_augment.split('_')]
            self.augment = RandAugment(num_ops, magnitude, num_magnitude_bins)

        if self.split == 'train':
            self.cur_tfm = VideoTransform(crop_size=self.crop_size, scale=self.scale, 
                ratio=self.aspect_ratio, eval_tfm=False)
        elif self.split == 'test':
           self.cur_tfm = VideoTransform(crop_size=self.crop_size, eval_tfm=True)
        else:
            raise NotImplementedError(f'Unknown split: {self.split}')


        self.pre_load = pre_load
        if self.pre_load:
            raise NotImplementedError('Pre-loading is not implemented yet')

        self.index_videos()

    def process_csv_data(self, csv_file, cls_num, vid_num):
        for i in range(10):
            try:
                csv_data = pd.read_csv(csv_file)
                if 'label' in csv_data:
                    if vid_num == -1:
                        vid_list = csv_data.sort_values(['label', 'path']).groupby('label', group_keys=False).apply(lambda x: x)
                    else:
                        vid_list = csv_data.sort_values(['label', 'path']).groupby('label', group_keys=False).head(vid_num)   

                    if cls_num != -1:
                        vid_list = pd.concat([group for _,group in vid_list.groupby('label')][:cls_num])

                    vid_list, _, _ = [vid_list[k].tolist() for k in ['path', 'label', 'action']]
                    self.vid_list += vid_list
                else:
                    self.vid_list += csv_data['path'].tolist()
                    
                return

            except Exception as e:
                print(e)
                if not os.path.exists(csv_file):
                    print(f'{csv_file} does not exist')
                    raise FileNotFoundError(f'{csv_file} does not exist')
                print(f'Error reading {csv_file}, retrying ({i+1}/5)...')
                print(f'{vid_num=}, {cls_num=}')
                print(f'{csv_data.size=}') # should be 38148
                print(f'{csv_data.info=}')
                print(f'{csv_data.columns.tolist()=}')
                continue

        raise RuntimeError(f'Failed to read {csv_file} after 10 retries (100s)')


    @property
    def is_master(self):
        return not dist.is_initialized() or dist.get_rank() == 0


    def index_videos(self):
        vid_list = self.vid_list
        if not self.multiple_datasets and Path(self.csv_file).stem.startswith('ucf'):
            actions = set()
            vid2action = {}
            for vid in vid_list:
                video_name = Path(vid).stem
                assert video_name.startswith('v_')
                action = video_name.split('_')[1]
                actions.add(action)
                vid2action[vid] = action

            actions = sorted(list(actions))
            assert len(actions) == 101, f'UCF101 has 101 classes, but got {len(actions)} classes'
            self.num_classes = len(actions)
            self.label2action = {i: actions[i] for i in range(len(actions))}
            self.action2label = {actions[i]: i for i in range(len(actions))}
            self.vid2label = {vid: self.action2label[vid2action[vid]] for vid in vid_list}

        if self.use_all_frames:
            preloaded = '_preloaded' if self.pre_load else ''
            cache_name = f'{self.csv_file}_{self.frame_num}_all_frames{preloaded}.pkl'
            cache_path = os.path.join(self.index_map_cache_dir, cache_name)

            if dist.is_initialized():
                dist.barrier()

            if self.is_master:
                if os.path.exists(cache_path):
                    print(f'Loading index map from {cache_path}')
                    with open(cache_path, 'rb') as f:
                        cached = pickle.load(f)
                        self.idx2label = cached['idx2label']
                        self.index_map = cached['index_map']
                else:
                    self.idx2label = dict()
                    index_map = {}
                    index = 0
                    for vid in tqdm(vid_list, desc='Indexing videos'):
                        vr = decord.VideoReader(vid)
                        n_groups = len(vr) // self.frame_num
                        for i in range(n_groups):
                            index_map[index] = (vid, i*self.frame_num, (i+1)*self.frame_num)
                            self.idx2label[index] = self.vid2label[vid]
                            index += 1
                    self.index_map = index_map

                    cached = {'idx2label': self.idx2label, 'index_map': self.index_map}
                    with open(cache_path, 'wb') as f:
                        pickle.dump(cached, f)

            if dist.is_initialized():
                dist.barrier()

            if not self.is_master:
                assert os.path.exists(cache_path), f'Failed to find {cache_path}'
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    self.idx2label = cached['idx2label']
                    self.index_map = cached['index_map']

        else:
            self.idx2label = {i: self.vid2label[vid] for i, vid in enumerate(vid_list)}

        if self.num_classes is not None:
            all_labels = list(self.idx2label.values())
            try:
                self.label_count = [all_labels.count(label) for label in range(self.num_classes)]
            except:
                self.label_count = None

            assert set(all_labels) == set(
                range(self.num_classes)
            ), f'Labels should be 0-{self.num_classes-1}, but got {set(all_labels)}'

            # count the number each unique label, store in a dictionary
            self.label_count = [all_labels.count(label) for label in range(self.num_classes)]
        else:
            self.label_count = None

    def __len__(self):
        if self.use_all_frames:
            length = len(self.index_map)
        else:
            length = len(self.vid_list)
        return length

    def get_video_batch_from_disk(self, idx):
        if self.fake:
            return torch.randint(0, 256, (self.frame_num, self.crop_size, self.crop_size, 3), dtype=torch.uint8), 'fake_path'

        if self.use_all_frames:
            vid, start, end = self.index_map[idx]
            vr = decord.VideoReader(vid)
            frame_idx = list(range(start, end))
            path = vid
        else:
            # Loading video
            vr = read_video_with_retry(self.vid_list[idx])
            frame_num = min(self.frame_num, len(vr))
            if self.frame_rate == 'uniform':
                frame_idx = [int(x*len(vr)/frame_num) for x in range(frame_num)]
            elif self.frame_rate == 'native':
                starting_idx = np.random.randint(0, len(vr) - frame_num + 1)
                frame_idx = list(range(starting_idx, starting_idx + frame_num))
            else:
                raise ValueError(f'Unknown frame_rate setting: {self.frame_rate}')
            path = self.vid_list[idx]

        video = vr.get_batch(frame_idx)
        return video, path

    def __getitem__(self, idx):
        video, path = self.get_video_batch_from_disk(idx)

        if self.augment is not None:
            video = self.augment(video.permute(0,-1,1,2)).permute(0,2,3,1)
        video = video.permute(-1,0,1,2).float() / 255. # # T,H,W,C -> C,T,H,W

        video_data = self.cur_tfm(video)
        if video.shape[1] < self.frame_num:
            video_data = F.pad(video_data, (0,0,0,0,0,self.frame_num-video.shape[1]), mode='replicate')

        label = self.idx2label[idx] if isinstance(self.idx2label[idx], int) else -1
        return {'gt': video_data, 'path': path, 'label': label}


