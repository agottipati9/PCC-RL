import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.utils import read_json_file


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Plot heat maps.")
    parser.add_argument('--reward-level', type=str, required=True,
                        choices=('mi', 'pkt'), help='Reward level')
    parser.add_argument('--heuristic', type=str, required=True,
                        choices=('bbr', 'cubic', 'bbr_old'),
                        help='Rule-based congestion control name.')
    parser.add_argument('--rl', type=str, required=True,
                        choices=('pretrained', 'genet_bbr', 'genet_cubic',
                                 'genet_bbr_old'),
                        help='Rule-based congestion control name.')
    parser.add_argument('--models-path', type=str,
                        help="path to genet trained Aurora models.")
    # parser.add_argument('--config-file', type=str, required=True,
    #                     help="path to configuration file.")
    parser.add_argument('--root', type=str, required=True,
                        help="path all exp results.")
    parser.add_argument('--dims', type=str, required=True, nargs=2,
                        choices=('bandwidth_lower_bound', 'delay', 'loss',
                                 'bandwidth_upper_bound', 'queue', 'T_s',
                                 'duration', 'delay_noise'),
                        help="2 dimenstions used to compare. Others use default values.")
    # parser.add_argument('--seed', type=int, default=42, help='seed')
    # parser.add_argument('--nproc', type=int, default=8, help='proc cnt')

    args, _ = parser.parse_known_args()
    return args


def find_idx(target_val, vals):
    for i in range(len(vals) - 1):
        if vals[i] <= target_val <= vals[i+1]:
            return i


def get_dim_vals(dim: str):
    if dim == 'bandwidth_upper_bound':
        dim_vals = [0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
                    7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0]
        dim_ticks = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17]
        dim_ticklabels = ['0.6', '1.0', '2.0', '3.0',
                          '5.0', '10.0', '20.0', '40.0', '80.0', '100.0']
        dim_axlabel = 'Max Bandwidth (Mbps)'
    elif dim == 'delay':
        dim_vals = [2, 5, 8, 10, 20, 50, 80, 100, 120, 150, 180, 200]
        dim_ticks = [0, 2, 4, 6, 8, 10]
        dim_ticklabels = ['4', '16', '40', '160', '240', '360']
        dim_axlabel = 'Latency (ms)'
    elif dim == 'loss':
        dim_vals = [0, 0.0001, 0.0002, 0.0005, 0.0008,
                    0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.05]
        dim_ticks = [0, 2, 4, 6, 8, 10]
        dim_ticklabels = ['0', '0.02', '0.08', '0.2', '0.8', '2']
        dim_axlabel = 'Random packet loss (%)'
    elif dim == 'queue':
        dim_vals = [0.1, 0.2, 0.5, 0.8, 1, 1.2, 1.6, 1.8, 2, 2.2, 2.5, 2.8, 3]
        dim_ticks = [0, 2, 4, 6, 8, 10, 12]
        dim_ticklabels = ['0.1', '0.5', '1', '1.6', '2', '2.5', '3']
        dim_axlabel = 'Queue (BDP)'
    elif dim == 'T_s':
        dim_vals = [0.1, 0.4, 0.6, 0.8, 1.2, 1.6,
                    2.0, 5, 8, 10.0, 12.0, 15.0, 17.0, 25, 30]
        dim_ticks = [0, 3, 6, 9, 12, 14]
        dim_ticklabels = ['0.1', '0.8', '2.0', '10.0', '17.0', '30.0']
        dim_axlabel = 'Bandwidth Change Interval (s)'
    else:
        raise NotImplementedError
    return dim_vals, dim_ticks, dim_ticklabels, dim_axlabel


def main():
    args = parse_args()
    dim0_vals, dim0_ticks, dim0_ticklabels, dim0_axlabel = get_dim_vals(args.dims[0])
    dim1_vals, dim1_ticks, dim1_ticklabels, dim1_axlabel = get_dim_vals(args.dims[1])
    fig, axes = plt.subplots(2, 5, figsize=(12, 10))
    max_gap = np.NINF
    min_gap = np.inf

    gap_matrices = []
    for bo in range(0, 30, 3):
        results = []
        for i in range(len(dim0_vals)):
            row = []
            for j in range(len(dim1_vals)):
                gaps = []
                for k in range(10):
                    trace_dir = os.path.join(
                        args.root, "{}_vs_{}/pair_{}_{}/trace_{}".format(args.dims[0], args.dims[1], i, j, k))
                    df = pd.read_csv(os.path.join(
                        trace_dir, args.heuristic, '{}_summary.csv'.format(args.heuristic)))
                    heuristic_reward = df['{}_level_reward'.format(
                        args.reward_level)]
                    if args.rl == 'pretrained':
                        df = pd.read_csv(os.path.join(
                            trace_dir, 'pretrained', 'aurora_summary.csv'))
                    else:
                        df = pd.read_csv(os.path.join(trace_dir, args.rl, 'seed_42', "bo_{}".format(
                            bo), 'step_64800', 'aurora_summary.csv'))
                    genet_reward = df['{}_level_reward'.format(
                        args.reward_level)]
                    gaps.append(genet_reward - heuristic_reward)
                row.append(np.mean(gaps))
                max_gap = max(max_gap, np.mean(gaps))
                min_gap = min(min_gap, np.mean(gaps))
            results.append(row)
        results = np.array(results)
        print(results.shape)
        gap_matrices.append(results)

    for subplot_idx, (gap_matrix, bo, ax) in enumerate(zip(gap_matrices, range(0, 30, 3), axes.flatten())):
        im = ax.imshow(gap_matrix)
        im.set_clim(vmax=0, vmin=min_gap)
        selected_configs = read_json_file(os.path.join(
            args.models_path, 'bo_{}.json'.format(bo)))

        selected_dim1_idxs = []
        selected_dim0_idxs = []

        if args.rl != 'pretrained':
            for selected_config in selected_configs[1:]:
                selected_dim1_idxs.append(
                    find_idx(selected_config[args.dims[1]][0], dim1_vals))
                selected_dim0_idxs.append(
                    find_idx(selected_config[args.dims[0]][0], dim0_vals))

            ax.scatter(selected_dim1_idxs,
                       selected_dim0_idxs, marker='o', c='r')
        if subplot_idx == 0 or subplot_idx == 5:
            ax.set_yticks(dim0_ticks)
            ax.set_yticklabels(dim0_ticklabels)
            ax.set_ylabel(dim0_axlabel)
        else:
            ax.set_yticks([])
        ax.set_xticks(dim1_ticks)
        ax.set_xticklabels(dim1_ticklabels)
        if subplot_idx == 2 or subplot_idx == 7:
            ax.set_xlabel(dim1_axlabel)

        if args.rl == 'pretrained':
            ax.set_title("pretrained")
            # plt.savefig(os.path.join(args.root, '{}_vs_{}'.format(args.dims[0], args.dims[1]),
            #                          '{}_{}_{}_level_reward_heatmap.jpg'.format(args.rl, args.heuristic, args.reward_level)))
            break
        else:
            ax.set_title("BO {}".format(bo))
            # plt.savefig(os.path.join(args.root, '{}_vs_{}'.format(args.dims[0], args.dims[1]),
            #                          '{}_{}_bo_{}_{}_level_reward_heatmap.jpg'.format(args.rl, args.heuristic, bo, args.reward_level)))
    cbar = fig.colorbar(im, ax=axes, location='bottom')
    cbar.ax.set_xlabel("{} - {}".format(args.rl, args.heuristic), rotation=0)
    # fig.tight_layout()
    plt.savefig(os.path.join(args.root, '{}_vs_{}'.format(args.dims[0], args.dims[1]),
                             '{}_{}_{}_level_reward_heatmap.jpg'.format(
                                 args.rl, args.heuristic, args.reward_level)), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
