import csv
import itertools
import os
import multiprocessing as mp
from typing import List

import numpy as np
from numba import jit

from simulator.abr_simulator.abr_trace import AbrTrace
from simulator.abr_simulator.schedulers import TestScheduler
from simulator.abr_simulator.env import Environment
from simulator.abr_simulator.constants import (
        A_DIM, B_IN_MB, DEFAULT_QUALITY, M_IN_K, MILLISECONDS_IN_SECOND,
        VIDEO_BIT_RATE, VIDEO_CHUNK_LEN, REBUF_PENALTY, SMOOTH_PENALTY,
        BUFFER_NORM_FACTOR, TOTAL_VIDEO_CHUNK)
# from simulator.abr_simulator.utils import linear_reward


S_LEN = 8

MPC_FUTURE_CHUNK_COUNT = 5

CHUNK_COMBO_OPTIONS = np.array([combo for combo in itertools.product(
                range(len(VIDEO_BIT_RATE)), repeat=MPC_FUTURE_CHUNK_COUNT)])
past_errors = []
past_bandwidth_ests = []

RANDOM_SEED = 42


@jit(nopython=True)
def get_chunk_size(quality, index, size_video_array):
    if index < 0 or index > TOTAL_VIDEO_CHUNK:
        return 0
    # note that the quality and video labels are inverted (i.e., quality 4 is
    # highest and this pertains to video1)
    return size_video_array[quality, index]

@jit(nopython=True)
def calculate_rebuffer(size_video_array, future_chunk_length, buffer_size,
                       bit_rate, last_index, future_bandwidth, bitrate_options):
    max_reward = -100000000
    start_buffer = buffer_size

    for full_combo in CHUNK_COMBO_OPTIONS:
        combo = full_combo[0:future_chunk_length]
        # calculate total rebuffer time for this combination (start with start_buffer and subtract
        # each download time and add 2 seconds in that order)
        curr_rebuffer_time = 0
        curr_buffer = start_buffer
        bitrate_sum = 0
        smoothness_diffs = 0
        last_quality = int( bit_rate )
        for position in range( 0, len( combo ) ):
            chunk_quality = combo[position]
            # e.g., if last chunk is 3, then first iter is 3+0+1=4
            index = last_index + position + 1
            # this is MB/MB/s --> seconds
            download_time = (get_chunk_size(chunk_quality, index, size_video_array) / 1000000.) / future_bandwidth
            if (curr_buffer < download_time):
                curr_rebuffer_time += (download_time - curr_buffer)
                curr_buffer = 0
            else:
                curr_buffer -= download_time
            curr_buffer += VIDEO_CHUNK_LEN / MILLISECONDS_IN_SECOND
            bitrate_sum += bitrate_options[chunk_quality]
            smoothness_diffs += abs(
                bitrate_options[chunk_quality] - bitrate_options[last_quality])
            last_quality = chunk_quality

        reward = (bitrate_sum / 1000.) - (REBUF_PENALTY *
                                          curr_rebuffer_time) - (smoothness_diffs / 1000.)
        if reward >= max_reward:
            best_combo = combo
            max_reward = reward
            send_data = 0
            if best_combo.size != 0:  # some combo was good
                send_data = best_combo[0]
    return send_data


class RobustMPC(object):
    abr_name = "mpc"


    def test_traces(self, traces: List[AbrTrace], video_size_file_dir: str, save_dir: str):
        rewards = []
        for trace in traces:
            rewards.append(self.test(trace, video_size_file_dir, save_dir))
        return rewards

    def test(self, trace: AbrTrace, video_size_file_dir: str, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        abr_log = open(os.path.join(save_dir, "{}_log.csv".format(self.abr_name)), 'w')
        log_writer = csv.writer(abr_log)
        log_writer.writerow(["timestamp", "bitrate", "buffer_size",
                             "rebuffering", "video_chunk_size", "delay",
                             "reward"])

        np.random.seed(RANDOM_SEED)

        # all_cooked_time ,all_cooked_bw ,all_file_names = load_traces(self.test_dir)

        # net_env = example_env_config(args, all_cooked_time, all_cooked_bw)
        trace_scheduler = TestScheduler(trace)
        net_env = Environment(trace_scheduler, VIDEO_CHUNK_LEN / MILLISECONDS_IN_SECOND,
                              video_size_file_dir=video_size_file_dir)
        size_video_array = np.array([net_env.video_size[i] for i in
                                     sorted(net_env.video_size)])
        # example_env_config(args, all_cooked_time, all_cooked_bw)

        # log_path = os.path.join(
        #     summary_dir, 'log_sim_mpc_' + all_file_names[net_env.trace_idx])
        # log_file = open(log_path, 'w', 1)

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((MPC_FUTURE_CHUNK_COUNT, S_LEN))]
        a_batch = [action_vec]
        r_batch = []

        # video_count = 0
        final_reward = 0

        # make chunk combination options
        # for combo in itertools.product([0, 1, 2, 3, 4, 5], repeat=5):
        #     CHUNK_COMBO_OPTIONS.append(combo)

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                          VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
            r_batch.append(reward)

            last_bit_rate = bit_rate

            # log time_stamp, bit_rate, buffer_size, reward
            # log_file.write(str(time_stamp / M_IN_K) + '\t' +
            #                str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
            #                str(buffer_size) + '\t' +
            #                str(rebuf) + '\t' +
            #                str(video_chunk_size) + '\t' +
            #                str(delay) + '\t' +
            #                str(reward) + '\n')

            log_writer.writerow([time_stamp / M_IN_K, VIDEO_BIT_RATE[bit_rate],
                                 buffer_size, rebuf, video_chunk_size, delay,
                                 reward])

            # retrieve previous state
            # if len(s_batch) == 0:
            #     state = [np.zeros((MPC_FUTURE_CHUNK_COUNT, S_LEN))]
            # else:
            state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = rebuf
            state[3, -1] = float(video_chunk_size) / \
                float(delay) / M_IN_K  # kilo byte / ms
            state[4, -1] = np.minimum(video_chunk_remain, TOTAL_VIDEO_CHUNK) / \
                    float(TOTAL_VIDEO_CHUNK)
            # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

            # ================== MPC =========================
            curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
            if (len(past_bandwidth_ests) > 0):
                curr_error = abs(
                    past_bandwidth_ests[-1]-state[3, -1])/float(state[3, -1])
            past_errors.append(curr_error)

            # pick bitrate according to MPC
            # first get harmonic mean of last 5 bandwidths
            past_bandwidths = state[3, -5:]
            while past_bandwidths[0] == 0.0:
                past_bandwidths = past_bandwidths[1:]

            bandwidth_sum = 0
            for past_val in past_bandwidths:
                bandwidth_sum += (1/float(past_val))
            harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

            error_pos = -5
            if (len(past_errors) < 5):
                error_pos = -len(past_errors)
            max_error = float(max(past_errors[error_pos:]))
            future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
            past_bandwidth_ests.append(harmonic_bandwidth)

            # future chunks length (try 4 if that many remaining)
            last_index = int(TOTAL_VIDEO_CHUNK - video_chunk_remain)
            future_chunk_length = min(MPC_FUTURE_CHUNK_COUNT, TOTAL_VIDEO_CHUNK - last_index)

            # all possible combinations of 5 chunk bitrates (9^5 options)
            # iterate over list and for each, compute reward and store max reward combination
            # start = time.time()
            #chunk_combo_options = np.array( CHUNK_COMBO_OPTIONS )

            bit_rate = calculate_rebuffer(size_video_array, future_chunk_length, buffer_size, bit_rate,
                                          last_index, future_bandwidth, np.array(VIDEO_BIT_RATE))

            s_batch.append(state)

            if end_of_video:
                # log_file.write('\n')
                # log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                # print(r_batch, len(r_batch))
                final_reward = sum(r_batch)
                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((MPC_FUTURE_CHUNK_COUNT, S_LEN)))
                a_batch.append(action_vec)

                # video_count += 1

                # if video_count >= len( all_file_names ):
                #     break

                # log_path = os.path.join(
                #     summary_dir,
                #     'log_sim_mpc_' + all_file_names[net_env.trace_idx])
                # log_file = open(log_path, 'w', 1)
                break
        abr_log.close()
        return final_reward
