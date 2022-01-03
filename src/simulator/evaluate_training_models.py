"""Evaluate all training models and rule-based methods on a set of real traces.
Used to generate Figure 12.
"""
import argparse
import os

import pandas as pd

import numpy as np

from simulator.aurora import test_on_traces
from simulator.network_simulator.bbr import BBR
from simulator.network_simulator.bbr_old import BBR_old
from simulator.network_simulator.cubic import Cubic
# from simulator.network_simulator.pcc.vivace.vivace_latency import VivaceLatency
from simulator.pantheon_dataset import PantheonDataset

# UDR1_ROOT = '/Users/zxxia/Project/PCC-RL/models/udr_small_lossless/seed_20'
# UDR2_ROOT = '/Users/zxxia/Project/PCC-RL/models/udr_mid_lossless/seed_20'
# UDR3_ROOT = '/Users/zxxia/Project/PCC-RL/models/udr_large_lossless/seed_20'
# UDR1_ROOT = '../results/resmodels/udr_2/udr_small_seed_20/seed_20'
UDR_ROOT = '../../results_0826/udr_6'
UDR_ROOT = '../../results_0826/udr_7'
UDR3_ROOT = '/Users/zxxia/Project/PCC-RL/models/udr_2/udr_large/seed_20'
# GENET_MODEL_PATH = "test/bo_3/model_step_72000.ckpt"
GENET_MODEL_PATH = "../../models/udr_large_lossless/seed_20/model_step_2124000.ckpt"
GENET_MODEL_PATH = "../../models/udr_large_lossless/seed_20/model_step_2124000.ckpt"
# GENET_MODEL_PATH = "test/bo_3/model_step_72000.ckpt"
# GENET_MODEL_PATH = "../../models/bo_10_model_step_36000/bo_10_model_step_36000.ckpt"


RESULT_ROOT = "../../results_0826"
RESULT_ROOT = "../../results_0910"
TRACE_ROOT = "../../data"
# TRACE_ROOT = "../../data/cellular/2018-12-11T00-27-AWS-Brazil-2-to-Colombia-cellular-3-runs-3-flows"

TARGET_CCS = ["bbr", "cubic", "vegas", "indigo", "ledbat", "quic"]
# TARGET_CCS = ['bbr'] #, "cubic", "vegas", "indigo", "ledbat", "quic"]


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Aurora Testing in simulator.")
    parser.add_argument('--save-dir', type=str, default="",
                        help="direcotry to testing results.")
    parser.add_argument('--cc', type=str, required=True,
                        choices=("bbr", 'bbr_old', "cubic", "udr1", "udr2", "udr3",
                                 "genet_bbr", 'genet_bbr_old', 'genet_cubic', 'real'),
                        help='congestion control name')
    parser.add_argument("--conn-type", type=str, required=True,
                        choices=('ethernet', 'cellular', 'wifi'),
                        help='connection type')
    parser.add_argument('--models-path', type=str, default="",
                        help="path to Aurora models.")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--nproc', type=int, default=16, help='proc cnt')

    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    if args.conn_type == 'ethernet':
        queue_size = 500
    elif args.conn_type == 'cellular':
        queue_size = 50
    else:
        raise ValueError
    dataset = PantheonDataset(TRACE_ROOT, args.conn_type, post_nsdi=False,
                              target_ccs=TARGET_CCS)
    traces = dataset.get_traces(0, queue_size, front_offset=10, wrap=True)
    save_dirs = [os.path.join(args.save_dir, args.conn_type, link_name,
                              trace_name) for link_name, trace_name in dataset.trace_names]

    if args.cc == 'bbr':
        cc = BBR(False)
        cc.test_on_traces(traces, [os.path.join(save_dir, cc.cc_name)
                                   for save_dir in save_dirs], True, args.nproc)
    elif args.cc == 'bbr_old':
        cc = BBR_old(False)
        cc.test_on_traces(traces, [os.path.join(save_dir, cc.cc_name)
                                   for save_dir in save_dirs], True, args.nproc)
    elif args.cc == 'cubic':
        cc = Cubic(False)
        cc.test_on_traces(traces, [os.path.join(save_dir, cc.cc_name)
                                   for save_dir in save_dirs], True, args.nproc)
    elif args.cc == 'real':
        model_path = args.models_path
        real_save_dirs = [os.path.join(
            save_dir, args.cc, "seed_{}".format(args.seed)) for save_dir in save_dirs]
        test_on_traces(model_path, traces, real_save_dirs,
                       args.nproc, 42, False, plot_flag=True)
    elif args.cc == 'udr1' or args.cc == 'udr2' or args.cc == 'udr3':
        # TODO: bug here when there is no validation log
        # original implementation
        # val_log = pd.read_csv(os.path.join(
        #     args.models_path, 'validation_log.csv'), sep='\t')
        # for idx in np.linspace(0, len(val_log['num_timesteps']) / 2 - 1, 10):
        #     step = int(val_log['num_timesteps'].iloc[int(idx)])
        #     udr_seed = ''
        #     for s in args.models_path.split('/'):
        #         if 'seed' in s:
        #             udr_seed = s
        #     udr_save_dirs = [os.path.join(
        #         save_dir, args.cc, udr_seed, "step_{}".format(step)) for save_dir in save_dirs]
        #     model_path = os.path.join(
        #         args.models_path, 'model_step_{}.ckpt'.format(step))
        #     test_on_traces(model_path, traces, udr_save_dirs,
        #                    args.nproc, 42, False, False)
        # new implementation to be fair with genet
        udr_seed = ''
        for s in args.models_path.split('/'):
            if 'seed' in s:
                udr_seed = s
        step = 64800
        while True:
            if not os.path.exists(os.path.join(args.models_path, 'model_step_{}.ckpt.meta'.format(step))):
                break
            udr_save_dirs = [os.path.join(
                save_dir, args.cc, udr_seed, "step_{}".format(step)) for save_dir in save_dirs]
            model_path = os.path.join(
                args.models_path, 'model_step_{}.ckpt'.format(step))
            test_on_traces(model_path, traces, udr_save_dirs,
                           args.nproc, 42, False, True)
            step += 64800
    elif args.cc == 'genet_bbr' or args.cc == 'genet_cubic' or 'genet_bbr_old':
        genet_seed = ''
        for s in args.models_path.split('/'):
            if 'seed' in s:
                genet_seed = s
        for bo in range(0, 15, 3):
            bo_dir = os.path.join(args.models_path, "bo_{}".format(bo))
            step = 64800
            model_path = os.path.join(
                bo_dir, 'model_step_{}.ckpt'.format(step))
            if not os.path.exists(model_path + '.meta'):
                continue
            genet_save_dirs = [os.path.join(
                save_dir, args.cc, genet_seed, "bo_{}".format(bo),
                "step_{}".format(step)) for save_dir in save_dirs]
            # print(genet_save_dirs)
            test_on_traces(model_path, traces, genet_save_dirs,
                           args.nproc, 42, False, True)
    else:
        raise ValueError


if __name__ == "__main__":
    main()
