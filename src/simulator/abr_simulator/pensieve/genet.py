import argparse
import subprocess
# import numpy as np
import os
from bayes_opt import BayesianOptimization
from simulator.abr_simulator.abr_trace import generate_trace
from simulator.abr_simulator.utils import map_log_to_lin, latest_actor_from, map_to_unnormalize
from simulator.abr_simulator.mpc import RobustMPC
from simulator.abr_simulator.pensieve.pensieve import Pensieve
# Default
TOTAL_EPOCHS = 100000
BAYESIAN_OPTIMIZER_INTERVAL = 5000
NUM_BO_UPDATE = int(TOTAL_EPOCHS / BAYESIAN_OPTIMIZER_INTERVAL)

TRAINING_DATA_DIR = "../data/BO_stable_traces/train/"
VAL_TRACE_DIR = '../data/BO_stable_traces/test/val_FCC/'
RESULTS_DIR = "../BO-results/randomize-param"

# TODO: the param switching is still manual now
PARAM_TYPE = "Trace"    #  or "Env"
# CURRENT_PARAM = "MAX_THROUGHPUT"

CURRENT_PARAM = "MIN_THROUGHPUT"
CURRENT_PARAM_MIN = 0.2
CURRENT_PARAM_MAX = 5

# CURRENT_PARAM = "BW_FREQ"
# CURRENT_PARAM_MIN = 2
# CURRENT_PARAM_MAX = 100

############################################################
# PARAM_TYPE = "Env"

# CURRENT_PARAM = "LINK_RTT"
# CURRENT_PARAM_MIN = 20
# CURRENT_PARAM_MAX = 1000

# CURRENT_PARAM = "BUFFER_THRES"
# CURRENT_PARAM_MIN = 2
# CURRENT_PARAM_MAX = 100
#
# CURRENT_PARAM = "CHUNK_LEN"
# CURRENT_PARAM_MIN = 1
# CURRENT_PARAM_MAX = 10

def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("BO training in simulator.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="directory to save testing and intermediate results.")
    parser.add_argument('--model-path', type=str, default=None,
                        help="path to Aurora model to start from.")
    parser.add_argument("--config-file", type=str,
                        help="Path to configuration file.")
    parser.add_argument("--bo-rounds", type=int, default=30,
                        help="Rounds of BO.")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--heuristic', type=str, default="mpc",
                        choices=('mpc'),
                        help='Congestion control rule based method.')

    return parser.parse_args()

def black_box_function(min_bw, max_bw, bw_change_interval, link_rtt,
                       buffer_thresh, duration, heuristic, model_path):
    '''
    :param x: input is the current params
    :return: reward is the mpc-rl reward
    '''
    traces = [generate_trace(bw_change_interval, duration, min_bw, max_bw,
                             link_rtt, buffer_thresh) for _ in range(10)]
    # path = os.path.join( RESULTS_DIR, 'model_saved' )
    # latest_model_path = latest_actor_from(path)

    #TODO: different param need different mapping
    # if CURRENT_PARAM == "MAX_THROUGHPUT":
    #     x_map = map_log_to_lin(x)
    # else:
    #     x_map = map_to_unnormalize(x, min_value=CURRENT_PARAM_MIN, max_value=CURRENT_PARAM_MAX)

    # print(x_map, "-----current value")

    # command = " python rl_test.py  \
    #             --param_name={updating_param_name} \
    #             --CURRENT_VALUE={current_test_value} \
    #             --test_trace_dir='../data/example_traces/' \
    #             --summary_dir='../MPC_RL_test_results/' \
    #             --model_path='{model_path}' \
    #             ".format(updating_param_name=CURRENT_PARAM, current_test_value=x_map, model_path=latest_model_path)
    #
    # r = float(subprocess.check_output(command, shell=True, text=True).strip())
    # return r

class Genet:

    def train(self, rounds: int):
        """
        """
        # BO guided training flow:
        for i in range(0, rounds):
            pbounds = {'x': (0 ,1)}
            optimizer = BayesianOptimization(
                f=black_box_function ,
                pbounds=pbounds
            )

            optimizer.maximize(
                init_points=13,
                n_iter=2,
                kappa=20,
                xi=0.1
            )
            next = optimizer.max
            param = next.get( 'params' ).get( 'x' )
            #bo_best_param = round( param ,2 )
            bo_best_param = map_log_to_lin(param)
            print( "BO chose this best param........", param, bo_best_param )

            # # Use the new param, add more traces into Pensieve, train more based on before

            # bo_best_param = 100   # for debugging
            path = os.path.join( RESULTS_DIR ,'model_saved' )
            latest_model_path = latest_actor_from( path )

            command = "python multi_agent.py \
                            --TOTAL_EPOCH=10000\
                            --train_trace_dir={training_dir} \
                            --val_trace_dir='{val_dir}'\
                            --summary_dir={results_dir}\
                            --description='first-run' \
                            --nn_model={model_path}\
                            --param_type_flag={param_type} \
                            --param_name={param_name} \
                            --CURRENT_VALUE={bo_output_param}"  \
                            .format(training_dir=TRAINING_DATA_DIR, val_dir=VAL_TRACE_DIR,
                                    results_dir=RESULTS_DIR, model_path=latest_model_path,
                                    param_type=PARAM_TYPE, param_name=CURRENT_PARAM, bo_output_param=bo_best_param)
            os.system(command)

            print("Running training:", i)
            i += 1

        print("Hooray!")

def main():
    args = parse_args()

    if args.heuristic == 'mpc':
        heuristic = RobustMPC()
    else:
        raise NotImplementedError


    genet = Genet()

    genet.train(args.bo_rounds, )



if __name__ == '__main__':
    main()
