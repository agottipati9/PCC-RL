from typing import List, Tuple

import numpy as np

from common import sender_obs
from common.utils import pcc_aurora_reward
from simulator.network_simulator.constants import (
    BITS_PER_BYTE, BYTES_PER_PACKET, MAX_RATE, MI_RTT_PROPORTION, MIN_RATE, TCP_INIT_CWND, MIN_CWND)
from simulator.network_simulator.sender import Sender
from simulator.network_simulator import packet
from simulator.trace import Trace

BTLBW_FILTER_LEN = 10  # packet-timed round trips.

class RateSample:
    def __init__(self):

        # The delivery rate sample (in most cases rs.delivered / rs.interval).
        self.delivery_rate = 0.0
        # The P.is_app_limited from the most recent packet delivered; indicates
        # whether the rate sample is application-limited.
        self.is_app_limited = False
        # The length of the sampling interval.
        self.interval = 0.0
        # The amount of data marked as delivered over the sampling interval.
        self.delivered = 0
        # The P.delivered count from the most recent packet delivered.
        self.prior_delivered = 0
        # The P.delivered_time from the most recent packet delivered.
        self.prior_time = 0.0
        # Send time interval calculated from the most recent packet delivered
        # (see the "Send Rate" section above).
        self.send_elapsed = 0.0
        # ACK time interval calculated from the most recent packet delivered
        # (see the "ACK Rate" section above).
        self.ack_elapsed = 0.0
        # in flight before this ACK
        self.prior_in_flight = 0
        # number of packets marked lost upon ACK
        self.losses = 0
        self.pkt_in_fast_recovery_mode = False


class AuroraBtlBwFilter:
    def __init__(self, btlbw_filter_len: int):
        self.btlbw_filter_len = btlbw_filter_len
        self.cache = {}

    def update(self, delivery_rate: float, round_count: int) -> None:
        self.cache[round_count] = max(self.cache.get(round_count, 0), delivery_rate)
        if len(self.cache) > self.btlbw_filter_len:
            self.cache.pop(min(self.cache))

    def get_btlbw(self) -> float:
        if not self.cache:
            return 0
        return max(self.cache.values())

    def reset(self):
        self.cache = {}


class AuroraPacket(packet.Packet):

    def __init__(self, ts: float, sender: Sender, pkt_id: int):
        super().__init__(ts, sender, pkt_id)
        self.delivered = 0
        self.delivered_time = 0.0
        self.first_sent_time = 0.0

    def debug_print(self):
        print("Event {}: ts={}, type={}, dropped={}, cur_latency: {}, "
              "delivered={}, delivered_time={}, first_sent_time={}, "
              "pkt_in_flight: {}, pacing_rate={}".format(
                  self.pkt_id, self.ts, self.event_type, self.dropped,
                  self.cur_latency, self.delivered, self.delivered_time,
                  self.first_sent_time,
                  self.sender.bytes_in_flight / BYTES_PER_PACKET, self.sender.pacing_rate / BYTES_PER_PACKET))


class AuroraSender(Sender):

    def __init__(self, pacing_rate: float, features: List[str],
                 history_len: int, sender_id: int, dest: int, trace: Trace):
        super().__init__(sender_id, dest)
        self.starting_rate = pacing_rate
        self.pacing_rate = pacing_rate
        self.pacing_rate = pacing_rate
        self.history_len = history_len
        self.features = features
        sender_obs._conn_min_latencies = {}
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.sender_id)
        self.trace = trace
        self.got_data = False
        self.cwnd = TCP_INIT_CWND
        self.min_latency = None
        self.prev_rtt_samples = []
        self.round_start = False
        self.round_count = 0
        self.next_round_delivered = 0
        self.delivered = 0

        self.btlbw_filter = AuroraBtlBwFilter(BTLBW_FILTER_LEN)
        # self.rs = RateSample()

    # # Upon receiving ACK, fill in delivery rate sample rs.
    # def generate_rate_sample(self, pkt: BBRPacket):
    #     # for each newly SACKed or ACKed packet P:
    #     #     self.update_rate_sample(P, rs)
    #     # fix the btlbw overestimation bug by not updating delivery_rate
    #     if not self.update_rate_sample(pkt):
    #         return False
    #
    #     # Clear app-limited field if bubble is ACKed and gone.
    #     if self.conn_state.app_limited and self.conn_state.delivered > self.conn_state.app_limited:
    #         self.app_limited = 0
    #
    #     # TODO: comment out and need to recheck
    #     # if self.rs.prior_time == 0:
    #     #     return False  # nothing delivered on this ACK
    #
    #     # Use the longer of the send_elapsed and ack_elapsed
    #     self.rs.interval = max(self.rs.send_elapsed, self.rs.ack_elapsed)
    #     # print(self.rs.send_elapsed, self.rs.ack_elapsed)
    #
    #     self.rs.delivered = self.conn_state.delivered - self.rs.prior_delivered
    #     # print("C.delivered: {}, rs.prior_delivered: {}".format(self.delivered, self.rs.prior_delivered))
    #
    #     # Normally we expect interval >= MinRTT.
    #     # Note that rate may still be over-estimated when a spuriously
    #     # retransmitted skb was first (s)acked because "interval"
    #     # is under-estimated (up to an RTT). However, continuously
    #     # measuring the delivery rate during loss recovery is crucial
    #     # for connections suffer heavy or prolonged losses.
    #
    #     if self.rs.interval < self.rtprop:
    #         self.rs.interval = -1
    #         return False  # no reliable sample
    #     self.rs.pkt_in_fast_recovery_mode = pkt.in_fast_recovery_mode
    #     if self.rs.interval != 0 and not pkt.in_fast_recovery_mode:
    #         self.rs.delivery_rate = self.rs.delivered / self.rs.interval
    #         # if self.rs.delivery_rate * 8 / 1e6 > 1.2:
    #         # print("C.delivered:", self.conn_state.delivered, "rs.prior_delivered:", self.rs.prior_delivered, "rs.delivered:", self.rs.delivered, "rs.interval:", self.rs.interval, "rs.delivery_rate:", self.rs.delivery_rate * 8 / 1e6)
    #
    #     return True  # we filled in rs with a rate sample
    #
    # # Update rs when packet is SACKed or ACKed.
    # def update_rate_sample(self, pkt: BBRPacket):
    #     # comment out because we don't need this in the simulator.
    #     # if pkt.delivered_time == 0:
    #     #     return  # P already SACKed
    #
    #     self.rs.prior_in_flight = self.bytes_in_flight
    #     self.conn_state.delivered += pkt.pkt_size
    #     self.conn_state.delivered_time = self.get_cur_time()
    #
    #     # Update info using the newest packet:
    #     # print(pkt.delivered, self.rs.prior_delivered)
    #     # if pkt.delivered > self.rs.prior_delivered:
    #     if (not self.rs.prior_delivered) or pkt.delivered > self.rs.prior_delivered:
    #         self.rs.prior_delivered = pkt.delivered
    #         self.rs.prior_time = pkt.delivered_time
    #         self.rs.is_app_limited = pkt.is_app_limited
    #         self.rs.send_elapsed = pkt.sent_time - pkt.first_sent_time
    #         self.rs.ack_elapsed = self.conn_state.delivered_time - pkt.delivered_time
    #         # print("pkt.sent_time:", pkt.sent_time, "pkt.first_sent_time:", pkt.first_sent_time, "send_elapsed:", self.rs.send_elapsed)
    #         # print("C.delivered_time:", self.conn_state.delivered_time, "P.delivered_time:", pkt.delivered_time, "ack_elapsed:", self.rs.ack_elapsed)
    #         self.conn_state.first_sent_time = pkt.sent_time
    #         return True
    #     return False
    #     # pkt.debug_print()
    #
    #     # Mark the packet as delivered once it's SACKed to
    #     # avoid being used again when it's cumulatively acked.
    #
    #     # pkt.delivered_time = 0

    def can_send_packet(self):
        return self.bytes_in_flight < self.cwnd * BYTES_PER_PACKET

    def on_packet_sent(self, pkt: AuroraPacket) -> bool:
        # if self.bytes_in_flight / BYTES_PER_PACKET == 0:
        #     self.conn_state.first_sent_time = self.get_cur_time()
        #     self.conn_state.delivered_time = self.get_cur_time()
        # pkt.first_sent_time = self.conn_state.first_sent_time
        # pkt.delivered_time = self.conn_state.delivered_time
        self.schedule_send()
        if self.can_send_packet():
            pkt.delivered = self.delivered
            super().on_packet_sent(pkt)
            return True
        return False

    def on_packet_acked(self, pkt: AuroraPacket) -> None:
        self.min_latency = min(self.min_latency, pkt.rtt) if self.min_latency else pkt.rtt
        self.delivered += pkt.pkt_size
        # self.conn_state.delivered_time = self.get_cur_time()
        super().on_packet_acked(pkt)
        # print(pkt.delivered, self.next_round_delivered)
        self.rtt_samples_ts.append(self.get_cur_time())
        if not self.got_data:
            self.got_data = len(self.rtt_samples) >= 1
        if (pkt.delivered == 0 and self.next_round_delivered == 0):
            self.round_start = False
        elif pkt.delivered >= self.next_round_delivered:
            self.next_round_delivered = self.delivered
            self.round_count += 1
            self.round_start = True
        else:
            self.round_start = False
        # self.generate_rate_sample(pkt)

    def on_packet_lost(self, pkt: "packet.Packet") -> None:
        return super().on_packet_lost(pkt)

    def apply_rate_delta(self, delta):
        if delta >= 0.0:
            self.set_rate(self.pacing_rate * (1.0 + float(delta)))
        else:
            self.set_rate(self.pacing_rate / (1.0 - float(delta)))

    def set_rate(self, new_rate):
        self.pacing_rate = new_rate
        if self.pacing_rate > MAX_RATE * BYTES_PER_PACKET:
            self.pacing_rate = MAX_RATE * BYTES_PER_PACKET
        if self.pacing_rate < MIN_RATE * BYTES_PER_PACKET:
            self.pacing_rate = MIN_RATE * BYTES_PER_PACKET

    def record_run(self):
        smi = self.get_run_data()
        self.history.step(smi)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        obs_end_time = self.get_cur_time()

        if not self.rtt_samples and self.prev_rtt_samples:
            rtt_samples = [np.mean(self.prev_rtt_samples)]
        else:
            rtt_samples = self.rtt_samples
        # rtt_samples is empty when there is no packet acked in MI
        # Solution: inherit from previous rtt_samples.

        # recv_start = self.rtt_samples_ts[0] if len(
        #     self.rtt_samples) >= 2 else self.obs_start_time
        recv_start = self.history.back().recv_end if len(
            self.rtt_samples) >= 1 else self.obs_start_time
        recv_end = self.rtt_samples_ts[-1] if len(
            self.rtt_samples) >= 1 else obs_end_time
        bytes_acked = self.acked * BYTES_PER_PACKET
        if recv_start == 0:
            recv_start = self.rtt_samples_ts[0]
            bytes_acked = (self.acked - 1) * BYTES_PER_PACKET

        # bytes_acked = max(0, (self.acked-1)) * BYTES_PER_PACKET if len(
        #     self.rtt_samples) >= 2 else self.acked * BYTES_PER_PACKET
        return sender_obs.SenderMonitorInterval(
            self.sender_id,
            bytes_sent=self.sent * BYTES_PER_PACKET,
            # max(0, (self.acked-1)) * BYTES_PER_PACKET,
            # bytes_acked=self.acked * BYTES_PER_PACKET,
            bytes_acked=bytes_acked,
            bytes_lost=self.lost * BYTES_PER_PACKET,
            send_start=self.obs_start_time,
            send_end=obs_end_time,
            # recv_start=self.obs_start_time,
            # recv_end=obs_end_time,
            recv_start=recv_start,
            recv_end=recv_end,
            rtt_samples=rtt_samples,
            queue_delay_samples=self.queue_delay_samples,
            packet_size=BYTES_PER_PACKET
        )

    def stop_run(self, pkt: "packet.Packet", end_time: float)->bool:
        ret = self.round_start
        self.round_start = False

        return ret #= True
        # return self.got_data and pkt.ts >= end_time and pkt.event_type == EVENT_TYPE_SEND

    def schedule_send(self, first_pkt: bool = False, on_ack: bool = False):
        assert self.net, "network is not registered in sender."
        if first_pkt:
            next_send_time = 0.0
        else:
            next_send_time = self.get_cur_time() + BYTES_PER_PACKET / self.pacing_rate
        next_pkt = AuroraPacket(next_send_time, self, 0)
        self.net.add_packet(next_pkt)

    def on_mi_start(self):
        self.reset_obs()

    def on_mi_finish(self) -> Tuple[float, float]:
        self.record_run()

        sender_mi = self.history.back()  # get_run_data()
        throughput = sender_mi.get("recv rate")  # bits/sec
        latency = sender_mi.get("avg latency")  # second
        loss = sender_mi.get("loss ratio")
        reward = pcc_aurora_reward(
            throughput / BITS_PER_BYTE / BYTES_PER_PACKET, latency, loss,
            self.trace.avg_bw * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET,
            self.trace.avg_delay * 2 / 1e3)

        if latency > 0.0:
            self.mi_duration = MI_RTT_PROPORTION * \
                sender_mi.get("avg latency") # + np.mean(extra_delays)
        self.btlbw_filter.update(throughput, self.round_count)
        min_lat = sender_mi.get("conn min latency")
        btlbw = self.btlbw_filter.get_btlbw()
        self.cwnd = max(2 * round(btlbw * self.min_latency / BITS_PER_BYTE / BYTES_PER_PACKET), MIN_CWND * 2)
        return reward, self.mi_duration

    def reset_obs(self):
        self.sent = 0
        self.acked = 0
        self.lost = 0
        if self.rtt_samples:
            self.prev_rtt_samples = self.rtt_samples
        self.rtt_samples = []
        self.rtt_samples_ts = []
        self.queue_delay_samples = []
        self.obs_start_time = self.get_cur_time()

    def reset(self):
        self.pacing_rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        sender_obs._conn_min_latencies = {}
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.sender_id)
        self.estRTT = 1000000 / 1e6  # SynInterval in emulation
        self.RTTVar = self.estRTT / 2  # RTT variance

        self.got_data = False
        self.prev_rtt_samples = []
        self.delivered = 0
        self.next_round_delivered = 0
        self.round_start = False
        self.round_count = 0
        self.cwnd = TCP_INIT_CWND
        self.btlbw_filter.reset()
