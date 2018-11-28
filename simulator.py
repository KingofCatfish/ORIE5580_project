__title__ = 'simulator'
__version__ = '1.0.1'
__author__ = 'Dajun Luo'

from heapq import heappush, heappop
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


class Transaction(object):
    """
    Represents a single transaction.
    """
    def __init__(self, amount, fee, create_time):
        """
        :param amount: transaction amount
        :param fee: transaction fee
        :param create_time: create time
        """
        self.amount = amount
        self.fee = fee
        self.create_time = create_time
        self.complete_time = None

    def complete(self, complete_time):
        """
        Mark the transaction is completed
        :param complete_time: complete time
        :return:
        """
        self.complete_time = complete_time

    def is_complete(self):
        """
        Returns a boolean of whether the transaction is finished
        :return:
        """
        return self.complete_time is not None


class Transcation_Pool(object):
    """
    A pool of all transaction in the simulation system. We use a heap structure to retrieve the transaction with
    highest fee.
    """
    def __init__(self, K):
        self.pool = []
        self.K = K
        self.pending_transaction = 0
        self.finished_transaction = 0
        self.pending_amount = 0
        self.finished_amount = 0
        self.pending_fee = 0
        self.finished_fee = 0
        self.delay = 0
        self.block = 0

    def add_transaction(self, transaction):
        """
        Add one transaction into the poll
        :param transaction: transaction object
        :return: None
        """
        self.pending_transaction += 1
        self.pending_amount += transaction.amount
        self.pending_fee += transaction.fee
        heappush(self.pool, (-transaction.fee, transaction))

    def generate_block(self, time):
        """
        Generate a block
        :param time: the time of generation
        :return: None
        """
        self.block += 1
        for i in range(self.K):
            if len(self.pool) == 0:
                return
            fee, transaction = heappop(self.pool)
            transaction.complete(time)
            self.pending_transaction -= 1
            self.pending_amount -= transaction.amount
            self.pending_fee -= transaction.fee
            self.finished_transaction += 1
            self.finished_amount += transaction.amount
            self.finished_fee += transaction.fee
            self.delay += time - transaction.create_time


class BRC_log(object):
    """
    Records stats information.
    """
    def __init__(self):
        self.time_line = defaultdict(list)

    def snapshot(self, simulator):
        """
        Take a snapshot of the current status in the simulation system
        :param simulator: simulator object
        :return: None
        """
        self.time_line['time(minutes)'].append(simulator.time)
        pool = simulator.pool
        self.time_line['all_transaction_count'].append(pool.pending_transaction + pool.finished_transaction)
        self.time_line['pending_transaction_count'].append(pool.pending_transaction)
        self.time_line['finished_transaction_count'].append(pool.finished_transaction)
        self.time_line['all_transaction_amount'].append(pool.pending_amount + pool.finished_amount)
        self.time_line['pending_transaction_amount'].append(pool.pending_amount)
        self.time_line['finished_transaction_amount'].append(pool.finished_amount)
        self.time_line['all_transaction_fee'].append(pool.pending_fee + pool.finished_fee)
        self.time_line['pending_transaction_fee'].append(pool.pending_fee)
        self.time_line['finished_transaction_fee'].append(pool.finished_fee)
        self.time_line['delay'].append(pool.delay)
        self.time_line['block'].append(pool.block)

    def generate_stats(self):
        """
        Generate overall stats information, such as average, from time line.
        :return: A pandas data frame of stats information.
        """
        data = pd.DataFrame(self.time_line)
        data['average_delay'] = data['delay'] / data['finished_transaction_count']
        data['average_transaction_amount'] = data['all_transaction_amount'] / data['all_transaction_count']
        data['average_transaction_fee'] = data['all_transaction_fee'] / data['all_transaction_count']
        data['average_pending_transaction_amount'] = data['pending_transaction_amount'] / data[
            'pending_transaction_count']
        data['average_pending_transaction_fee'] = data['pending_transaction_fee'] / data['pending_transaction_count']
        data['average_finished_transaction_amount'] = data['finished_transaction_amount'] / data[
            'finished_transaction_count']
        data['average_finished_transaction_fee'] = data['finished_transaction_fee'] / data['finished_transaction_count']
        data['average_block_size'] = data['finished_transaction_count'] / data['block']
        data['average_block_amount'] = data['finished_transaction_amount'] / data['block']
        data['average_block_fee'] = data['finished_transaction_fee'] / data['block']
        return data


class Simulator(object):
    """
    Simulator of the dynamic in the DES system. We use a heap transaction to fast retrieve the next event.
    """
    def __init__(self, transaction_generator, mu_generator, K):
        """
        Configure the simulator
        :param transaction_generator: a function returns a random transaction (amount, fee, time interval).
        :param mu_generator: a function return a random time interval for block generation.
        :param K: maximum transaction in a block
        """
        self.transaction_generator = transaction_generator
        self.mu_generator = mu_generator
        self.K = K
        self.reset()

    def add_transaction_event(self):
        # Generate a transaction
        amount, fee, inter_arrival_time = self.transaction_generator()
        # Add it into the transaction pool
        self.pool.add_transaction(Transaction(amount, fee, self.time))
        # Add next transaction into the event pool
        heappush(self.event, (self.time + inter_arrival_time, self.add_transaction_event))

    def generate_block_event(self):
        # Generate a block
        self.pool.generate_block(self.time)
        # Add next block generation event into the event pool
        heappush(self.event, (self.time + self.mu_generator(), self.generate_block_event))

    def run(self, duration, warm_up=0):
        """
        Run simulation.
        :param duration: length of the simulation
        :param warm_up: warm up time
        :return: None
        """
        # Loop until there is no event
        while len(self.event) != 0:
            time, event = heappop(self.event)
            if time > duration:
                # Stop simulation after reach the deadline
                break
            # Update the time stamp
            self.time = time
            event()
            if self.time > warm_up:
                # Log after warmed up
                self.log.snapshot(self)
        # Generate statistics
        self.stats = self.log.generate_stats()

    def reset(self):
        """
        Reset the simulation system.
        :return: None
        """
        self.event = [(0, self.add_transaction_event), (0, self.generate_block_event)]
        self.pool = Transcation_Pool(self.K)
        self.time = 0
        self.log = BRC_log()
        self.stats = None

    def plot(self, response):
        """
        Plot response versus time.
        :param response: available response
        :return: seaborn plot
        """
        if response not in self.stats.columns:
            raise ValueError('Only the following variables can be plotted. %s' % str(self.stats.columns))
        plt.figure(figsize=(20,10))
        plt.plot(self.stats['time(minutes)'], self.stats[response])
        plt.xlabel('time (minutes)')
        plt.ylabel(response)

