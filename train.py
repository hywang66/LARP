"""
    Generate a cfg object according to a cfg file and args, then spawn Trainer(rank, cfg).
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from easydict import EasyDict as edict
from mergedeep import merge

import trainers
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    parser.add_argument('--data_path', default='data/k400')
    parser.add_argument('--csv_file', default='k400_train.js')
    parser.add_argument('--eval_frames', type=str, default='none')
    parser.add_argument('--frame_num', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--num_workers', '-j', type=int, default=16)
    parser.add_argument('--out_path', type=str, default='default')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--wandb-upload', '-w', action='store_true')
    parser.add_argument('--wandn_entity', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument(
        '--opts', type=str, nargs='*', default=[], help='cfg args to update'
    )
    parser.add_argument('--manualSeed', type=int, default=-1, help='manual seed')
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--debug', action='store_true')


    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    return args


def make_cfg(args):
    if args.debug:
        args.name = 'debug'
        if args.wandb_upload:
            print('!!!wandb upload is disabled in debug mode')
            args.wandb_upload = False
        args.replace = True

    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    def translate_cfg_(d):
        for k, v in d.items():
            if isinstance(v, dict):
                translate_cfg_(v)
            elif isinstance(v, str):
                if v.startswith('$') and v.endswith('$'):
                    v = getattr(args, v.replace('$', ''))
                d[k] = v

    translate_cfg_(cfg)

    if args.name is None:
        exp_name = os.path.basename(args.cfg).split('.')[0]
    else:
        exp_name = args.name

    env = edict()
    env['tot_gpus'] = torch.cuda.device_count()
    env['cudnn'] = args.cudnn
    env['wandb_upload'] = args.wandb_upload
    if args.wandn_entity is not None:
        env['wandb_entity'] = args.wandn_entity
    if args.wandb_project is not None:
        env['wandb_project'] = args.wandb_project 
    cfg['env'] = env

    def build_tree(tree_list):
        if len(tree_list) >= 2:
            return {
                tree_list[0]: (
                    build_tree(tree_list[1:]) if len(tree_list) > 2 else tree_list[-1]
                )
            }

    def nested_v(dict, keys):
        for key in keys:
            dict = dict[key]
        return dict

    def convert(type, x):
        if type == bool and isinstance(x, str):
            if x.lower() == 'true':
                return True
            elif x.lower() == 'false':
                return False
            else:
                raise ValueError('Cannot convert {} to bool'.format(x))
        elif (type == list or type == tuple) and isinstance(x, str):
            x = x.split('_')
            return [eval(x0) for x0 in x]
        else:
            return type(x)

    assert len(args.opts) % 2 == 0
    for cur_cfg_key, v in zip(args.opts[::2], args.opts[1::2]):
        keys = cur_cfg_key.split('.')
        v = convert(type(nested_v(cfg, keys)), v)
        cfg = merge(cfg, build_tree(keys + [v]))

    cfg = edict(cfg)
    cfg.comment = args.comment
    cfg.train_dataset.args.cls_vid_num = cfg.train_dataset.args.cls_vid_num.strip(
        "'"
    ).strip('"')

    env.exp_name = trainers.trainers_dict[cfg['trainer']].get_exp_name(
        exp_name, cfg, args
    )
    env.save_dir = os.path.join(args.out_path, env.exp_name)
    env.port = str(2960 + utils.hash_string_to_int(env.save_dir) % 10000)
    cfg.manualSeed = args.manualSeed
    return cfg


def main_worker(rank, cfg):
    manualSeed = cfg['manualSeed']
    if manualSeed != -1:
        manualSeed += rank
        torch.manual_seed(manualSeed)
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True

    if cfg['compile']:
        from einops._torch_specific import \
            allow_ops_in_compiled_graph  # requires einops>=0.6.1
        allow_ops_in_compiled_graph()

    trainer = trainers.trainers_dict[cfg['trainer']](rank, cfg)
    trainer.run()


def main():
    args = parse_args()
    cfg = make_cfg(args)
    utils.ensure_path(cfg['env']['save_dir'], args.replace)
    if cfg['env']['tot_gpus'] > 1:
        mp.spawn(main_worker, args=(cfg,), nprocs=cfg['env']['tot_gpus'])
    else:
        main_worker(0, cfg)


if __name__ == '__main__':
    main()