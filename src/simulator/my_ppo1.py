import os
import time
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from mpi4py import MPI

from stable_baselines import PPO1
from stable_baselines.common import Dataset, explained_variance, fmt_row, zipsame, SetVerbosity, \
    TensorboardWriter
from stable_baselines import logger
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.mpi_moments import mpi_moments
from stable_baselines.common.misc_util import flatten_lists
from stable_baselines.common.runners import traj_segment_generator
from stable_baselines.trpo_mpi.utils import add_vtarg_and_adv

class MyPPO1(PPO1):

    def predict(self, observation, state=None, mask=None, deterministic=False, saliency=False):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        grad = None
        if deterministic and saliency:
            actions, _, states, _, grad = self.step(observation, state, mask, deterministic=deterministic, saliency=saliency)
        else:
            actions, _, states, _ = self.step(observation, state, mask, deterministic=deterministic)

        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            clipped_actions = clipped_actions[0]

        if deterministic and saliency:
            return clipped_actions, states, grad
        elif deterministic and saliency:
            return clipped_actions, states
        else:
            return clipped_actions, states

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="PPO1",
              reset_num_timesteps=True):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO1 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            with self.sess.as_default():
                self.adam.sync()
                callback.on_training_start(locals(), globals())

                # Prepare for rollouts

                seg_gen = traj_segment_generator(self.policy_pi, self.env, self.timesteps_per_actorbatch,
                                                 callback=callback)

                episodes_so_far = 0
                timesteps_so_far = 0
                iters_so_far = 0
                t_start = time.time()

                # rolling buffer for episode lengths
                len_buffer = deque(maxlen=100)
                # rolling buffer for episode rewards
                reward_buffer = deque(maxlen=100)

                while True:
                    # print(timesteps_so_far)
                    if timesteps_so_far >= total_timesteps:
                        break

                    if self.schedule == 'constant':
                        cur_lrmult = 1.0
                    elif self.schedule == 'linear':
                        cur_lrmult = max(1.0 - float(timesteps_so_far) / total_timesteps, 0)
                    else:
                        raise NotImplementedError

                    logger.log("********** Iteration %i ************" % iters_so_far)

                    # self.env.save_dir = "/tank/zxxia/PCC-RL/results_0928/verify_alp_continuity_test/bw_delay/seed_42/train_traces/worker_{}_iteration_{}".format(
                    #         MPI.COMM_WORLD.Get_rank(), iters_so_far)
                    # self.env.save_dir = "/tank/zxxia/PCC-RL/results_0928/verify_alp_continuity_test/bw_delay2/seed_42/ref_traces/worker_{}_iteration_{}".format(
                    #         MPI.COMM_WORLD.Get_rank(), iters_so_far)
                    # os.makedirs(self.env.save_dir, exist_ok=True)

                    t_collection_start = time.time()
                    seg = seg_gen.__next__()

                    print("myppo1 collection:", time.time() - t_collection_start, )

                    # Stop training early (triggered by the callback)
                    if not seg.get('continue_training', True):  # pytype: disable=attribute-error
                        break

                    add_vtarg_and_adv(seg, self.gamma, self.lam)

                    # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                    observations, actions = seg["observations"], seg["actions"]
                    atarg, tdlamret = seg["adv"], seg["tdlamret"]

                    # true_rew is the reward without discount
                    if writer is not None:
                        total_episode_reward_logger(self.episode_reward,
                                                    seg["true_rewards"].reshape((self.n_envs, -1)),
                                                    seg["dones"].reshape((self.n_envs, -1)),
                                                    writer, self.num_timesteps)

                    # predicted value function before udpate
                    vpredbefore = seg["vpred"]

                    # standardized advantage function estimate
                    atarg = (atarg - atarg.mean()) / atarg.std()
                    dataset = Dataset(dict(ob=observations, ac=actions, atarg=atarg, vtarg=tdlamret),
                                      shuffle=not self.policy.recurrent)
                    optim_batchsize = self.optim_batchsize or observations.shape[0]
                    # np.set_printoptions(formatter={'float': "{0:0.2e}".format})
                    # for row in np.concatenate([observations, actions, atarg.reshape(-1, 1), tdlamret.reshape(-1, 1)], axis=1).tolist():
                    #     csv_writer.writerow(map(lambda t: "%0.2e" % t, row))

                    # set old parameter values to new parameter values
                    self.assign_old_eq_new(sess=self.sess)
                    logger.log("Optimizing...")
                    logger.log(fmt_row(13, self.loss_names))

                    # Here we do a bunch of optimization epochs over the data
                    for k in range(self.optim_epochs):
                        t_start = time.time()
                        # list of tuples, each of which gives the loss for a minibatch
                        losses = []
                        for i, batch in enumerate(dataset.iterate_once(optim_batchsize)):
                            steps = (self.num_timesteps +
                                     k * optim_batchsize +
                                     int(i * (optim_batchsize / len(dataset.data_map))))
                            if writer is not None:
                                # run loss backprop with summary, but once every 10 runs save the metadata
                                # (memory, compute time, ...)
                                if self.full_tensorboard_log and (1 + k) % 10 == 0:
                                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                    run_metadata = tf.RunMetadata()
                                    summary, grad, *newlosses = self.lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult, sess=self.sess,
                                                                                 options=run_options,
                                                                                 run_metadata=run_metadata)
                                    writer.add_run_metadata(run_metadata, 'step%d' % steps)
                                else:
                                    summary, grad, *newlosses = self.lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult, sess=self.sess)
                                writer.add_summary(summary, steps)
                            else:
                                _, grad, *newlosses = self.lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                       batch["atarg"], batch["vtarg"], cur_lrmult,
                                                                       sess=self.sess)

                            self.adam.update(grad, self.optim_stepsize * cur_lrmult)
                            losses.append(newlosses)
                        logger.log(fmt_row(13, np.mean(losses, axis=0)))
                        # print('model update per epoch: {}s'.format(time.time() - t_start))

                    logger.log("Evaluating losses...")
                    losses = []
                    for batch in dataset.iterate_once(optim_batchsize):
                        newlosses = self.compute_losses(batch["ob"], batch["ob"], batch["ac"], batch["atarg"],
                                                        batch["vtarg"], cur_lrmult, sess=self.sess)
                        losses.append(newlosses)
                    mean_losses, _, _ = mpi_moments(losses, axis=0)
                    logger.log(fmt_row(13, mean_losses))
                    for (loss_val, name) in zipsame(mean_losses, self.loss_names):
                        logger.record_tabular("loss_" + name, loss_val)
                    logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

                    # local values
                    lrlocal = (seg["ep_lens"], seg["ep_rets"])

                    # list of tuples
                    listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)
                    lens, rews = map(flatten_lists, zip(*listoflrpairs))
                    len_buffer.extend(lens)
                    reward_buffer.extend(rews)
                    if len(len_buffer) > 0:
                        logger.record_tabular("EpLenMean", np.mean(len_buffer))
                        logger.record_tabular("EpRewMean", np.mean(reward_buffer))
                    logger.record_tabular("EpThisIter", len(lens))
                    episodes_so_far += len(lens)
                    current_it_timesteps = MPI.COMM_WORLD.allreduce(seg["total_timestep"])
                    timesteps_so_far += current_it_timesteps
                    self.num_timesteps += current_it_timesteps
                    iters_so_far += 1
                    logger.record_tabular("EpisodesSoFar", episodes_so_far)
                    logger.record_tabular("TimestepsSoFar", self.num_timesteps)
                    logger.record_tabular("TimeElapsed", time.time() - t_start)
                    if self.verbose >= 1 and MPI.COMM_WORLD.Get_rank() == 0:
                        logger.dump_tabular()
        callback.on_training_end()
        return self
