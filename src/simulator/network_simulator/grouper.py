import random
import numpy as np

from simulator.network_sim import BYTES_PER_PACKET

class Grouper:
    """Group a sequence of packets to simulate bursty behavior.
    """
    def __init__(self, net, count: int, ratio=0.7):
        self.net = net
        self.buffer = []
        self.count = count
        self.ratio = ratio
        self.start = False
        self.n = 0
        self.total_num = 0
        self.start_ts = 0

    def group(self, pkt):
        if not self.start:
            rand = random.uniform(0, 1)
            if rand < 0.002:
            # if rand < 0:
                self.start = True
                self.start_ts = pkt.ts
            else:
                return True
        else:
            pkt.grouped = True
            self.buffer.append(pkt)
            # if len(self.buffer) == self.count:
            if 1000 * (pkt.ts - self.start_ts) > self.count:
                for pkt_tmp in self.buffer:
                    pkt_tmp.add_delay_noise(self.buffer[-1].ts - pkt_tmp.ts)
                    self.net.add_packet(pkt_tmp)
                self.buffer = []
                self.start = False
                self.start_ts = 0
                self.count = np.random.uniform(50, 280)

            return False
        pass
    def update(self, ts):
        if 1000 * (ts - self.start_ts) > self.count:
            for pkt_tmp in self.buffer:
                pkt_tmp.add_delay_noise(self.buffer[-1].ts - pkt_tmp.ts)
                self.net.add_packet(pkt_tmp)
            self.buffer = []
            self.start = False
            self.start_ts = 0
            self.count = np.random.uniform(50, 280)

    def group_bkp():
        return pkt
        if not self.start:
            rand = random.uniform(0, 1)
            if rand < 0.1:
            # if rand < 0:
                self.start = True
        else:
            pkt.add_delay_noise((self.count - self.n) * BYTES_PER_PACKET / pkt.pacing_rate * self.ratio)
            pkt.grouped = True
            self.n += 1
            if self.n == self.count:
                self.total_num += 1
                # print('changed', pkt.ts, self.total_num)
                self.n = 0
                self.start = False
        return pkt

    def reset(self):
        self.n = 0
        self.start = False
