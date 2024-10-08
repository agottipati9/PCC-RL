import argparse
import glob
import itertools
import os
import subprocess
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.utils import natural_sort, read_json_file, set_seed

from simulator.aurora import Aurora
from simulator.trace import generate_trace, generate_traces

matplotlib.use('Agg')


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Test UDR models in simulator.")
    # parser.add_argument('--exp-name', type=str, default="",
    #                     help="Experiment name.")
    parser.add_argument("--model-path", type=str, default=[], nargs="+",
                        help="Path to one or more Aurora Tensorflow checkpoints"
                        " or models to serve.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="direcotry to save the model.")
    parser.add_argument('--dimension', type=str,  # nargs=1,
                        choices=["bandwidth", "delay", "loss", "queue"])
    parser.add_argument('--config-file', type=str, required=True,
                        help="config file")
    parser.add_argument('--train-config-dir', type=str, default=None,
                        help="path to one or more training configs.")
    parser.add_argument('--duration', type=int, required=True,
                        help="trace duration")
    parser.add_argument('--delta-scale', type=float, required=True,
                        help="delta scale")
    parser.add_argument('--plot-only', action='store_true',
                        help='plot only if specified')
    parser.add_argument('--n-models', type=int, default=3,
                        help='Number of models to average on.')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    return parser.parse_args()


def multiple_runs(aurora_models, bw, delay, loss, queue, aurora_save_dir,
                  duration, plot_only):
    test_traces = [generate_trace((duration, duration), (bw, bw), (delay, delay),
                                  (loss, loss), (queue, queue))]

    rewards = []
    for aurora in aurora_models:
        aurora.log_dir = os.path.join(
            aurora_save_dir,
            os.path.splitext(os.path.basename(aurora.pretrained_model_path))[0])
        os.makedirs(aurora.log_dir, exist_ok=True)
        t_start = time.time()
        if not plot_only:
            aurora.test(test_traces)
        print("run {} on bw={}Mbps, delay={}ms, loss={}, "
              "queue={}packets, duration={}s, used {:.3f}s".format(
                  aurora.pretrained_model_path, bw, delay, loss, queue, duration,
                  time.time() - t_start))
        df = pd.read_csv(os.path.join(aurora.log_dir, "aurora_test_log.csv"))
        rewards.append(df['reward'].mean())
    return np.mean(np.array(rewards))


def get_last_n_models(model_path, n_models):
    parent_dir = os.path.dirname(model_path)
    ckpt_index_files = natural_sort(
        glob.glob(os.path.join(parent_dir, "model_step_*.ckpt.index")))
    target_model_ctime = os.path.getctime(model_path + ".index")
    ckpt_paths = []
    for ckpt_index_file in ckpt_index_files[::-1]:
        if os.path.getctime(ckpt_index_file) > target_model_ctime:
            continue
        ckpt_paths.append(os.path.splitext(ckpt_index_file)[0])
        if len(ckpt_paths) >= n_models:
            return ckpt_paths
    return ckpt_paths


def main():
    args = parse_args()
    set_seed(args.seed)
    metric = args.dimension
    model_paths = args.model_path
    save_root = args.save_dir
    config_file = args.config_file
    train_config_dir = args.train_config_dir
    print(metric, model_paths, save_root, config_file)

    plt.figure(figsize=(8, 8))
    config = read_json_file(config_file)
    print(config)
    bw_list = config['bandwidth']
    delay_list = config['delay']
    loss_list = config['loss']
    queue_list = config["queue"]

    # run cubic
    cubic_rewards = []
    # cubic_scores = []
    for (bw, delay, loss, queue) in itertools.product(
            bw_list, delay_list, loss_list, queue_list):
        save_dir = f"{save_root}/rand_{metric}/env_{bw}_{delay}_{loss}_{queue}"
        cubic_save_dir = os.path.join(save_dir, "cubic")
        if not args.plot_only:
            t_start = time.time()
            cmd = "python evaluate_cubic.py --bandwidth {} --delay {} " \
                "--loss {} --queue {} --save-dir {} --duration {} --seed {}".format(
                    bw, delay, loss, queue, cubic_save_dir, args.duration, args.seed)
            subprocess.check_output(cmd, shell=True).strip()
            print("run cubic on bw={}Mbps, delay={}ms, loss={}, "
                  "queue={}packets, duration={}s, used {:3f}s".format(
                      bw, delay, loss, queue, args.duration,
                      time.time() - t_start))
        df = pd.read_csv(os.path.join(
            cubic_save_dir, "cubic_test_log.csv"))
        cubic_rewards.append(df['reward'].mean())
        # cubic_scores.append(np.mean(learnability_objective_function(
        #     df['throughput'] * 1500 * 8 / 1e6, df['latency']*1000/2)))

    for model_idx, (model_path, ls, marker, color) in enumerate(
            zip(model_paths, ["-", "--", "-.", "-", "-", "--", "-.", ":"],
                ["x", "s", "v", '*', '+', '^', '>', '1'],
                ["C3", "C3", "C3", "C1", "C1", "C1", "C1", "C1"])):
        model_name = os.path.basename(os.path.dirname(model_path))
        aurora_rewards = []
        # aurora_scores = []
        # detect latest n models here
        last_n_model_paths = get_last_n_models(model_path, args.n_models)
        # construct n Aurora objects and load aurora models here
        last_n_auroras = []
        for tmp_model_path in last_n_model_paths:
            last_n_auroras.append(
                Aurora(seed=args.seed, log_dir="", timesteps_per_actorbatch=10,
                       pretrained_model_path=tmp_model_path,
                       delta_scale=args.delta_scale))

        for (bw, delay, loss, queue) in itertools.product(
                bw_list, delay_list, loss_list, queue_list):
            save_dir = f"{save_root}/rand_{metric}/env_{bw}_{delay}_{loss}_{queue}"
            aurora_save_dir = os.path.join(save_dir, model_name)
            cubic_save_dir = os.path.join(save_dir, "cubic")

            # run aurora
            aurora_rewards.append(
                multiple_runs(last_n_auroras, bw, delay, loss, queue,
                              aurora_save_dir, args.duration, args.plot_only))
        if model_idx == 0:
            plt.plot(config[metric], cubic_rewards, 'o-', c="C0",
                     label="TCP Cubic")
        if train_config_dir is not None:
            train_config = read_json_file(os.path.join(
                train_config_dir, model_name+'.json'))
            env = 'bw=[{}, {}]Mbps, delay=[{}, {}]ms, loss=[{}, {}], queue=[{}, {}]pkt'.format(
                train_config[0]['bandwidth'][0], train_config[0]['bandwidth'][1],
                train_config[0]['delay'][0], train_config[0]['delay'][1],
                train_config[0]['loss'][0], train_config[0]['loss'][1],
                train_config[0]['queue'][0], train_config[0]['queue'][1])
        else:
            env = ""
            # raise RuntimeError
        assert ls in {'', '-', '--', '-.', ':', None}
        plt.plot(config[metric], aurora_rewards, marker=marker,
                 linestyle=ls, c=color, label=model_name + ", " + env)
    plt.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.2), loc="lower left",
               mode="expand", ncol=1, )
    if metric == "bandwidth":
        unit = "Mbps"
    elif metric == 'delay':
        unit = 'ms'
    elif metric == 'loss':
        unit = ''
    elif metric == 'queue':
        unit = 'packets'
    else:
        raise RuntimeError

    plt.xlabel("{} ({})".format(metric, unit))
    plt.ylabel('Reward')
    # plt.ylabel('log(throughput) - log(delay)')
    plt.tight_layout()
    plt.savefig(os.path.join(
        args.save_dir, "rand_{}_sim.png".format(metric)))
    plt.close()
    # plt.show()


if __name__ == "__main__":
    main()
