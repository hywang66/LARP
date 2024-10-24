import hashlib
import logging
import os
import pickle
import shutil
import time

import numpy as np
import requests
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.utils.tensorboard import SummaryWriter


def ensure_path(path, replace=True):
    if os.path.exists(path):
        if replace:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_logger(file_path):
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel('INFO')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path, 'w')
    formatter = logging.Formatter('[%(asctime)s] %(message)s', '%m-%d %H:%M:%S')
    for handler in [stream_handler, file_handler]:
        handler.setFormatter(formatter)
        handler.setLevel('INFO')
        logger.addHandler(handler)
    return logger


def set_save_dir(save_dir, replace=True):
    ensure_path(save_dir, replace=replace)
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S')
    logger = set_logger(os.path.join(save_dir, f'log_{time_str}.txt'))
    writer = SummaryWriter(os.path.join(save_dir, f'tensorboard'))
    return logger, writer, time_str


def compute_num_params(model, full_model=True, text=True, exclude_loss=True):
    if full_model:
        params = set(model.parameters())
        if exclude_loss and hasattr(model, 'loss'):
            params -= set(model.loss.parameters())
        tot = int(sum([np.prod(p.shape) for p in params]))
    else:
        if not hasattr(model, 'base_params'):
            return 0
        tot = int(sum([v.nelement() for k,v in model.base_params.items()]))

    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        elif tot >= 1e3:
            return '{:.1f}K'.format(tot / 1e3)
        else:
            return str(tot)
    else:
        return tot


def text2str(tot):
    if tot >= 1e6:
        return '{:.1f}M'.format(tot / 1e6)
    elif tot >= 1e3:
        return '{:.1f}K'.format(tot / 1e3)
    else:
        return str(tot)


def make_optimizer(params, optimizer_spec, load_sd=False):
    optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW
    }[optimizer_spec['name']](params, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


class Averager():
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class EpochTimer():
    def __init__(self, max_epoch):
        self.max_epoch = max_epoch
        self.epoch = 0
        self.t_start = time.time()
        self.t_last = self.t_start

    def epoch_done(self):
        t_cur = time.time()
        self.epoch += 1
        epoch_time = t_cur - self.t_last
        tot_time = t_cur - self.t_start
        est_time = tot_time / self.epoch * self.max_epoch
        self.t_last = t_cur
        return time_text(epoch_time), time_text(tot_time), time_text(est_time)


def time_text(secs):
    if secs >= 3600:
        return f'{secs / 3600:.1f}h'
    elif secs >= 60:
        return f'{secs / 60:.1f}m'
    else:
        return f'{secs:.1f}s'


def split_str(s, sep='_', type=int):
    return [type(x) for x in s.split(sep) if x]


def hash_string_to_int(s):
    hash_object = hashlib.sha256(s.encode())
    hex_dig = hash_object.hexdigest()
    return int(hex_dig, 16)

def check_website_access_bool(url):
    try:
        response = requests.get(url, timeout=3)
        # Check if the response code is 200 or 403
        if response.status_code in [200, 403]:
            return True
        else:
            return False
    except (requests.exceptions.Timeout, requests.exceptions.RequestException):
        # Any exception means the site is not accessible within the parameters
        return False
    

def serialize_to_tensor(obj):
    obj_bytes = pickle.dumps(obj)
    obj_tensor = torch.ByteTensor(list(obj_bytes)).to('cuda')
    return obj_tensor, obj_tensor.size(0)


def deserialize_to_obj(obj_tensor):
    obj_bytes = bytes(obj_tensor.cpu().tolist())
    obj = pickle.loads(obj_bytes)
    return obj


def gather_object_from_all(py_object):
    """
    Gathers a Python object from all processes in a distributed environment.

    Args:
        py_object (Any): The Python object to be gathered.

    Returns:
        dict: A dictionary containing the gathered objects from all processes, where the keys are the ranks of the processes.

    Raises:
        AssertionError: If the distributed environment is not initialized.
        AssertionError: If the received object from any process is not equal to the original object, only happens if there is a bug in the code.

    """
    assert dist.is_initialized(), 'Distributed environment is not initialized'

    gathered_obj_dict = {}
    obj_tensor, obj_tensor_size = serialize_to_tensor(py_object)
    local_tensor_size_tensor = torch.tensor(obj_tensor_size).cuda()
    tensor_size_tensor = local_tensor_size_tensor.clone()
    dist.all_reduce(tensor_size_tensor, op=dist.ReduceOp.MAX)
    max_tensor_size = tensor_size_tensor.item()
    if obj_tensor_size < max_tensor_size:
        padding_size = max_tensor_size - obj_tensor_size
        local_obj_tensor_padded = F.pad(obj_tensor, (0, padding_size))
    else:
        local_obj_tensor_padded = obj_tensor.clone()

    for rank in range(dist.get_world_size()):
        received = local_obj_tensor_padded.clone()
        received_tensor_size = local_tensor_size_tensor.clone()
        dist.broadcast(received, src=rank)
        dist.broadcast(received_tensor_size, src=rank)
        received = received[:received_tensor_size.item()]
        if dist.get_rank() == rank:
            assert torch.equal(received, obj_tensor), f'Rank {rank} received object is not equal to the original object'
        gathered_obj_dict[rank] = deserialize_to_obj(received)

    return gathered_obj_dict


def repeat_to_m_frames(video_tensor, m=16):
    # video_tensor: (b, c, t, h, w)
    b, c, t, h, w = video_tensor.shape
    if t >= m:
        return video_tensor
    else:
        repeated_tensor = torch.cat([video_tensor, video_tensor[:, :, -1:].repeat(1, 1, m - t, 1, 1)], dim=2)
        return repeated_tensor