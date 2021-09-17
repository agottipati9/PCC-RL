import csv
import logging
import os
import shutil
import time
import types
from typing import List
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from mpi4py.MPI import COMM_WORLD
from mpi4py.futures import MPIPoolExecutor

import gym
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines import PPO1
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.results_plotter import load_results, ts2xy

from simulator import network
from simulator.constants import BYTES_PER_PACKET
from simulator.trace import generate_trace, Trace, generate_traces
from common.utils import set_tf_loglevel, pcc_aurora_reward
from plot_scripts.plot_packet_log import PacketLog, plot
from plot_scripts.plot_time_series import plot as plot_simulation_log
from udt_plugins.testing.loaded_agent import LoadedModel


if type(tf.contrib) != types.ModuleType:  # if it is LazyLoader
    tf.contrib._warning = None

set_tf_loglevel(logging.FATAL)


class MyMlpPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, **_kwargs):
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env,
                                          n_steps, n_batch, reuse, net_arch=[
                                              {"pi": [32, 16], "vf": [32, 16]}],
                                          feature_extraction="mlp", **_kwargs)

def val_on_trace(model_path, trace, save_dir):
    aurora = Aurora(20, "", 10, pretrained_model_path=model_path)
    result = aurora._test(trace, save_dir, plot_flag=False)
    print('finish test_on_trace')
    return result


def val_on_traces(model_path: str, traces: List[Trace], save_dirs: List[str]):
    arguments = [(model_path, trace, save_dir) for trace, save_dir in zip(traces, save_dirs)]
    with MPIPoolExecutor(max_workers=COMM_WORLD.Get_size()) as executor:
        return executor.starmap(val_on_trace, arguments)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using
    ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, aurora, check_freq: int, log_dir: str, val_traces: List = [],
                 verbose=0, steps_trained=0, config_file=None, validation_flag=False):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.aurora = aurora
        self.check_freq = check_freq
        self.log_dir = log_dir
        # self.save_path = os.path.join(log_dir, 'saved_models')
        self.save_path = log_dir
        self.best_mean_reward = -np.inf
        self.val_traces = val_traces
        self.config_file = config_file
        self.validation_flag = validation_flag
        if self.aurora.comm.Get_rank() == 0:
            self.val_log_writer = csv.writer(
                open(os.path.join(log_dir, 'validation_log.csv'), 'w', 1),
                delimiter='\t', lineterminator='\n')
            self.val_log_writer.writerow(
                ['n_calls', 'num_timesteps', 'mean_validation_reward', 'loss',
                 'throughput', 'latency', 'sending_rate', 'tot_t_used(min)',
                 'val_t_used(min)', 'train_t_used(min)'])

            os.makedirs(os.path.join(log_dir, "validation_traces"), exist_ok=True)
            for i, tr in enumerate(self.val_traces):
                tr.dump(os.path.join(log_dir, "validation_traces", "trace_{}.json".format(i)))
        else:
            self.val_log_writer = None
        self.best_val_reward = -np.inf
        self.val_times = 0

        self.t_start = time.time()
        self.prev_t = time.time()
        self.steps_trained = steps_trained

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            # x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            # if len(x) > 0:
            #     # Mean training reward over the last 100 episodes
            #     mean_reward = np.mean(y[-100:])
            #     if self.verbose > 0:
            #         print("Num timesteps: {}".format(self.num_timesteps))
            #         print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
            #             self.best_mean_reward, mean_reward))
            #
            #     # New best model, you could save the agent here
            #     if mean_reward > self.best_mean_reward:
            #         self.best_mean_reward = mean_reward
            #         # Example for saving best model
            #         if self.verbose > 0:
            #             print("Saving new best model to {}".format(self.save_path))
            #         # self.model.save(self.save_path)

            if self.aurora.comm.Get_rank() == 0 and self.val_log_writer is not None:
                model_path_to_save = os.path.join(self.save_path, "model_step_{}.ckpt".format(int(self.num_timesteps)))
                with self.model.graph.as_default():
                    saver = tf.train.Saver()
                    saver.save(self.model.sess, model_path_to_save)
                if not self.validation_flag:
                    return True
                avg_tr_bw = []
                avg_tr_min_rtt = []
                avg_tr_loss = []
                avg_rewards = []
                avg_losses = []
                avg_tputs = []
                avg_delays = []
                avg_send_rates = []
                val_start_t = time.time()

                save_dirs = [os.path.join(self.log_dir, "validation_traces", "trace_{}".format(i)) for i in range(len(self.val_traces))]
                for ts_list, val_rewards, loss_list, tput_list, delay_list, \
                        send_rate_list, action_list, obs_list, mi_list, pkt_log in val_on_traces(model_path_to_save, self.val_traces, save_dirs):
                    avg_rewards.append(np.mean(np.array(val_rewards)))
                    avg_losses.append(np.mean(np.array(loss_list)))
                    avg_tputs.append(float(np.mean(np.array(tput_list))))
                    avg_delays.append(np.mean(np.array(delay_list)))
                    avg_send_rates.append(
                        float(np.mean(np.array(send_rate_list))))

                # for idx, val_trace in enumerate(self.val_traces):
                #     avg_tr_bw.append(val_trace.avg_bw)
                #     avg_tr_min_rtt.append(val_trace.avg_bw)
                #     ts_list, val_rewards, loss_list, tput_list, delay_list, \
                #         send_rate_list, action_list, obs_list, mi_list, pkt_log = self.aurora._test(
                #             val_trace, self.log_dir)
                #     # pktlog = PacketLog.from_log(pkt_log)
                #     avg_rewards.append(np.mean(np.array(val_rewards)))
                #     avg_losses.append(np.mean(np.array(loss_list)))
                #     avg_tputs.append(float(np.mean(np.array(tput_list))))
                #     avg_delays.append(np.mean(np.array(delay_list)))
                #     avg_send_rates.append(
                #         float(np.mean(np.array(send_rate_list))))
                #     # avg_rewards.append(pktlog.get_reward())
                #     # avg_losses.append(pktlog.get_loss_rate())
                #     # avg_tputs.append(np.mean(pktlog.get_throughput()[1]))
                #     # avg_delays.append(np.mean(pktlog.get_rtt()[1]))
                #     # avg_send_rates.append(np.mean(pktlog.get_sending_rate()[1]))
                cur_t = time.time()
                self.val_log_writer.writerow(
                    map(lambda t: "%.3f" % t,
                        [float(self.n_calls), float(self.num_timesteps),
                         np.mean(np.array(avg_rewards)),
                         np.mean(np.array(avg_losses)),
                         np.mean(np.array(avg_tputs)),
                         np.mean(np.array(avg_delays)),
                         np.mean(np.array(avg_send_rates)),
                         (cur_t - self.t_start) / 60,
                         (cur_t - val_start_t) / 60, (val_start_t - self.prev_t) / 60]))
                self.prev_t = cur_t
        return True


def save_model_to_serve(model, export_dir):
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    with model.graph.as_default():

        pol = model.policy_pi  # act_model

        obs_ph = pol.obs_ph
        act = pol.deterministic_action
        sampled_act = pol.action

        obs_input = tf.saved_model.utils.build_tensor_info(obs_ph)
        outputs_tensor_info = tf.saved_model.utils.build_tensor_info(act)
        stochastic_act_tensor_info = tf.saved_model.utils.build_tensor_info(
            sampled_act)
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"ob": obs_input},
            outputs={"act": outputs_tensor_info,
                     "stochastic_act": stochastic_act_tensor_info},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        signature_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                         signature}

        model_builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        model_builder.add_meta_graph_and_variables(
            model.sess, tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_map,
            clear_devices=True)
        model_builder.save(as_text=True)


class Aurora():
    cc_name = 'aurora'
    def __init__(self, seed: int, log_dir: str, timesteps_per_actorbatch: int,
                 pretrained_model_path=None, gamma: float = 0.99,
                 tensorboard_log=None, delta_scale=1, record_pkt_log: bool = False):
        init_start = time.time()
        self.record_pkt_log = record_pkt_log
        self.comm = COMM_WORLD
        self.delta_scale = delta_scale
        self.seed = seed
        self.log_dir = log_dir
        self.pretrained_model_path = pretrained_model_path
        self.steps_trained = 0
        dummy_trace = generate_trace(
            (10, 10), (2, 2), (2, 2), (50, 50), (0, 0), (1, 1), (0, 0), (0, 0))
        env = gym.make('PccNs-v0', traces=[dummy_trace],
                       train_flag=True, delta_scale=self.delta_scale)
        # Load pretrained model
        # print('create_dummy_env,{}'.format(time.time() - init_start))
        if pretrained_model_path is not None:
            if pretrained_model_path.endswith('.ckpt'):
                model_create_start = time.time()
                self.model = PPO1(MyMlpPolicy, env, verbose=1, seed=seed,
                                  optim_stepsize=0.001, schedule='constant',
                                  timesteps_per_actorbatch=timesteps_per_actorbatch,
                                  optim_batchsize=int(
                                      timesteps_per_actorbatch/12),
                                  optim_epochs=12, gamma=gamma,
                                  tensorboard_log=tensorboard_log,
                                  n_cpu_tf_sess=1)
                # print('create_ppo1,{}'.format(time.time() - model_create_start))
                tf_restore_start = time.time()
                with self.model.graph.as_default():
                    saver = tf.train.Saver()
                    saver.restore(self.model.sess, pretrained_model_path)
                try:
                    self.steps_trained = int(os.path.splitext(
                        pretrained_model_path)[0].split('_')[-1])
                except:
                    self.steps_trained = 0
                # print('tf_restore,{}'.format(time.time()-tf_restore_start))
            else:
                # model is a tensorflow model to serve
                self.model = LoadedModel(pretrained_model_path)
        else:
            self.model = PPO1(MyMlpPolicy, env, verbose=1, seed=seed,
                              optim_stepsize=0.001, schedule='constant',
                              timesteps_per_actorbatch=timesteps_per_actorbatch,
                              optim_batchsize=int(timesteps_per_actorbatch/12),
                              optim_epochs=12, gamma=gamma,
                              tensorboard_log=tensorboard_log, n_cpu_tf_sess=1)
        self.timesteps_per_actorbatch = timesteps_per_actorbatch

    def train(self, config_file,
            # training_traces, validation_traces,
            total_timesteps, tot_trace_cnt,
              tb_log_name="", validaiton_flag=False):
        assert isinstance(self.model, PPO1)

        training_traces = generate_traces(config_file, tot_trace_cnt,
                                          duration=30)
        # generate validation traces
        validation_traces = generate_traces(
            config_file, 20, duration=30)
        env = gym.make('PccNs-v0', traces=training_traces,
                       train_flag=True, delta_scale=self.delta_scale, config_file=config_file)
        env.seed(self.seed)
        # env = Monitor(env, self.log_dir)
        self.model.set_env(env)

        # Create the callback: check every n steps and save best model
        callback = SaveOnBestTrainingRewardCallback(
            self, check_freq=self.timesteps_per_actorbatch, log_dir=self.log_dir,
            steps_trained=self.steps_trained, val_traces=validation_traces,
            config_file=config_file)
        self.model.learn(total_timesteps=total_timesteps,
                         tb_log_name=tb_log_name, callback=callback)

    def test_on_traces(self, traces: List[Trace], save_dirs: List[str]):
        results = []
        pkt_logs = []
        for trace, save_dir in zip(traces, save_dirs):
            ts_list, reward_list, loss_list, tput_list, delay_list, \
                send_rate_list, action_list, obs_list, mi_list, pkt_log = self._test(
                    trace, save_dir)
            result = list(zip(ts_list, reward_list, send_rate_list, tput_list,
                              delay_list, loss_list, action_list, obs_list, mi_list))
            pkt_logs.append(pkt_log)
            results.append(result)
        return results, pkt_logs

    def save_model(self, save_path):
        with self.model.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.model.sess, os.path.join(save_path, "tmp_model.ckpt"))
        return os.path.join(save_path, "tmp_model.ckpt")

    def load_model(self):
        raise NotImplementedError

    def _test(self, trace: Trace, save_dir: str, plot_flag=False):
        reward_list = []
        loss_list = []
        tput_list = []
        delay_list = []
        send_rate_list = []
        ts_list = []
        action_list = []
        mi_list = []
        obs_list = []
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            f_sim_log = open(os.path.join(save_dir, 'aurora_simulation_log.csv'), 'w', 1)
            writer = csv.writer(f_sim_log, lineterminator='\n')
            writer.writerow(['timestamp', "target_send_rate", "send_rate",
                             'recv_rate', 'max_recv_rate', 'latency',
                             'loss', 'reward', "action", "bytes_sent",
                             "bytes_acked", "bytes_lost", "MI",
                             "send_start_time",
                             "send_end_time", 'recv_start_time',
                             'recv_end_time', 'latency_increase',
                             "packet_size", 'min_lat', 'sent_latency_inflation',
                             'latency_ratio', 'send_ratio',
                             'bandwidth', "queue_delay",
                             'packet_in_queue', 'queue_size', 'cwnd',
                             'ssthresh', "rto", "recv_ratio", "srtt"])
        else:
            f_sim_log = None
            writer = None
        env = gym.make(
            'PccNs-v0', traces=[trace], delta_scale=self.delta_scale, record_pkt_log=self.record_pkt_log)
        env.seed(self.seed)
        obs = env.reset()
        pred_cost = 0
        step_cost = 0
        while True:
            pred_start = time.time()
            if isinstance(self.model, LoadedModel):
                obs = obs.reshape(1, -1)
                action = self.model.act(obs)
                action = action['act'][0]
            else:
                if env.net.senders[0].got_data:
                    action, _states = self.model.predict(
                        obs, deterministic=True)
                else:
                    action = np.array([0])
            pred_cost += time.time() - pred_start

            # get the new MI and stats collected in the MI
            # sender_mi = env.senders[0].get_run_data()
            sender_mi = env.senders[0].history.back() #get_run_data()
            # if env.net.senders[0].got_data:
            #     action = heuristic.step(obs, sender_mi)
            #     # action = my_heuristic.stateless_step(env.senders[0].send_rate,
            #     #         env.senders[0].avg_latency, env.senders[0].lat_diff, env.senders[0].start_stage,
            #     #         env.senders[0].max_tput, env.senders[0].min_rtt, sender_mi.rtt_samples[-1])
            #     # action = my_heuristic.stateless_step(*obs)
            # else:
            #     action = np.array([0])
            # max_recv_rate = heuristic.max_tput
            max_recv_rate = env.senders[0].max_tput
            throughput = sender_mi.get("recv rate")  # bits/sec
            send_rate = sender_mi.get("send rate")  # bits/sec
            latency = sender_mi.get("avg latency")
            loss = sender_mi.get("loss ratio")
            avg_queue_delay = sender_mi.get('avg queue delay')
            sent_latency_inflation = sender_mi.get('sent latency inflation')
            latency_ratio = sender_mi.get('latency ratio')
            send_ratio = sender_mi.get('send ratio')
            recv_ratio = sender_mi.get('recv ratio')
            reward = pcc_aurora_reward(
                throughput / 8 / BYTES_PER_PACKET, latency, loss,
                np.mean(trace.bandwidths) * 1e6 / 8 / BYTES_PER_PACKET, np.mean(trace.delays) * 2/ 1e3)
            if save_dir and writer:
                writer.writerow([
                    env.net.get_cur_time(), round(env.senders[0].rate * BYTES_PER_PACKET * 8, 0),
                    round(send_rate, 0), round(throughput, 0), round(max_recv_rate), latency, loss,
                    reward, action.item(), sender_mi.bytes_sent, sender_mi.bytes_acked,
                    sender_mi.bytes_lost, sender_mi.send_end - sender_mi.send_start,
                    sender_mi.send_start, sender_mi.send_end,
                    sender_mi.recv_start, sender_mi.recv_end,
                    sender_mi.get('latency increase'), sender_mi.packet_size,
                    sender_mi.get('conn min latency'), sent_latency_inflation,
                    latency_ratio, send_ratio,
                    env.links[0].get_bandwidth(
                        env.net.get_cur_time()) * BYTES_PER_PACKET * 8,
                    avg_queue_delay, env.links[0].pkt_in_queue, env.links[0].queue_size,
                    env.senders[0].cwnd, env.senders[0].ssthresh, env.senders[0].rto, recv_ratio, env.senders[0].estRTT])
            reward_list.append(reward)
            loss_list.append(loss)
            delay_list.append(latency * 1000)
            tput_list.append(throughput / 1e6)
            send_rate_list.append(send_rate / 1e6)
            ts_list.append(env.net.get_cur_time())
            action_list.append(action.item())
            mi_list.append(sender_mi.send_end - sender_mi.send_start)
            obs_list.append(obs.tolist())
            obs, rewards, dones, info = env.step(action)

            if dones:
                break
        if f_sim_log:
            f_sim_log.close()
        if self.record_pkt_log and save_dir:
            with open(os.path.join(save_dir, "aurora_packet_log.csv"), 'w', 1) as f:
                pkt_logger = csv.writer(f, lineterminator='\n')
                pkt_logger.writerow(['timestamp', 'packet_event_id', 'event_type',
                                     'bytes', 'cur_latency', 'queue_delay',
                                     'packet_in_queue', 'sending_rate', 'bandwidth'])
                pkt_logger.writerows(env.net.pkt_log)
        if self.record_pkt_log and plot_flag:
            pkt_log = PacketLog.from_log(env.net.pkt_log)
            plot(trace, pkt_log, save_dir, "aurora")
        if plot_flag and save_dir:
            plot_simulation_log(trace, os.path.join(save_dir, 'aurora_simulation_log.csv'), save_dir)

        tput = env.senders[0].tot_acked / (env.senders[0].last_ack_ts - env.senders[0].first_ack_ts)
        avg_lat = env.senders[0].cur_avg_latency
        loss = 1 - env.senders[0].tot_acked / env.senders[0].tot_sent
        pkt_level_reward = pcc_aurora_reward(tput, avg_lat,loss,
            avg_bw=trace.avg_bw * 1e6 / 8 / BYTES_PER_PACKET)

        return ts_list, reward_list, loss_list, tput_list, delay_list, send_rate_list, action_list, obs_list, mi_list, pkt_level_reward

    def test(self, trace: Trace, save_dir: str, plot_flag=False):
        _, reward_list, _, _, _, _, _, _, _, pkt_level_reward = self._test(trace, save_dir, plot_flag)
        return np.mean(reward_list), pkt_level_reward

def test_on_trace(model_path: str, trace: Trace, save_dir: str, seed: int):
    rl = Aurora(seed=seed, log_dir="", pretrained_model_path=model_path,
                timesteps_per_actorbatch=10)
    return rl.test(trace, save_dir, False)

def test_on_traces(model_path: str, traces: List[Trace], save_dirs: List[str], nproc: int, seed: int):
    arguments = [(model_path, trace, save_dir, seed) for trace, save_dir in zip(traces, save_dirs)]
    with MPIPoolExecutor(max_workers=nproc) as executor:
        results = executor.starmap(test_on_trace, arguments)
    return results
