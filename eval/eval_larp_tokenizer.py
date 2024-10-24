import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse

import torch

from eval.rfvd_evaluator import UCFrFVDEvaluator
from models.larp_tokenizer import LARPTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--dataset_csv', type=str, default='ucf101_train.csv')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--version', type=str, default='sd')
    parser.add_argument('--amp_dtype', type=str, default='float16')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--no_fvd', action='store_true')
    parser.add_argument('--det', action='store_true')
    parser.add_argument('--token_subsample', type=int, default=None)
    parser.add_argument('--repeat_to_16', action='store_true')
    return parser.parse_args()

def main(args):
    assert args.tokenizer is not None

    if os.path.exists(args.tokenizer):
        # Load tokenizer from local checkpoint
        model = LARPTokenizer.from_checkpoint(args.tokenizer)
    else:
        # Load tokenizer from HuggingFace Hub
        model = LARPTokenizer.from_pretrained(args.tokenizer)

    if args.det and hasattr(model, 'set_vq_eval_deterministic'):
        model.set_vq_eval_deterministic(deterministic=True)
        print('Using deterministic VQ for evaluation')
    
    if args.use_amp:
        if args.amp_dtype == 'float16':
            amp_dtype = torch.float16
        elif args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16
        else:
            raise ValueError(f'Unknown AMP dtype: {args.amp_dtype}')
    else:
        amp_dtype = torch.float32
    
    evaluator = UCFrFVDEvaluator(
        model,
        args.dataset_csv,
        use_amp=args.use_amp,
        amp_dtype=amp_dtype,
        compile=args.compile,
        frame_num=args.num_frames,
        batch_size=args.batch_size,
        num_workers=4,
        token_subsample=args.token_subsample,
        repeat_to_16=args.repeat_to_16,
    )
    mse, psnr_val, fvd, lpips_val = evaluator.evaluate(no_fvd=args.no_fvd)
    print(f'mse={mse}\npsnr_val={psnr_val}\nfvd={fvd}\nlpips_val={lpips_val}')


if __name__ == '__main__':
    args = get_args()
    main(args)



'''
# Usage:

python eval/eval_larp_tokenizer.py \
    --tokenizer hywang66/LARP-L-long-tokenizer \
    --use_amp --det

# Output:
# mse=0.0017696263967081904
# psnr_val=28.702985763549805
# fvd=19.53286303104324
# lpips_val=0.07546473294496536

'''