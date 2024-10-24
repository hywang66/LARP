import argparse
import glob
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import FeatureStats, FVDCalculator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_stats_dir', type=str, required=True)
    return parser.parse_args()

def load_feature_stats_from_multiple_files(files):
    feature_stats = FeatureStats.load(files[0])
    for file in files[1:]:
        feature_stats += FeatureStats.load(file)
    return feature_stats
    

def main(args):
    feature_stats_dir = args.feature_stats_dir
    generated_feature_stats_files = glob.glob(os.path.join(feature_stats_dir, 'generated_fvd_stats_*.pkl'))
    gt_feature_states_files = glob.glob(os.path.join(feature_stats_dir, 'gt_fvd_stats_*.pkl'))

    assert len(generated_feature_stats_files) == len(gt_feature_states_files) and len(generated_feature_stats_files) > 0

    print(f"Calculating FVD for {len(generated_feature_stats_files)} pairs of feature stats")
    fvd_calculator = FVDCalculator()

    generated_feature_stats = load_feature_stats_from_multiple_files(generated_feature_stats_files)
    gt_feature_stats = load_feature_stats_from_multiple_files(gt_feature_states_files)

    number_of_samples = generated_feature_stats.num_items
    print(f"Total umber of samples: {number_of_samples}")

    fvd = fvd_calculator.calculate_fvd(generated_feature_stats, gt_feature_stats)
    if isinstance(fvd, torch.Tensor):
        fvd = fvd.item()

    print(f"FVD: {fvd}")
    return fvd



if __name__ == '__main__':
    main(get_args())

 