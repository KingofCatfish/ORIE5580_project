__title__ = 'simulator'
__version__ = '1.1.0'
__author__ = 'Dajun Luo'

from heapq import heappush, heappop
from collections import defaultdict
import time as systime

import pandas as pd
import numpy as np
import seaborn as sns
sns.set(rc={'figure.figsize': (20, 10)})
from scipy import stats


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
        self.history = []
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
        heappush(self.pool, (-transaction.fee, id(transaction), transaction))
        self.history.append(transaction)

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
            fee, id, transaction = heappop(self.pool)
            transaction.complete(time)
            self.pending_transaction -= 1
            self.pending_amount -= transaction.amount
            self.pending_fee -= transaction.fee
            self.finished_transaction += 1
            self.finished_amount += transaction.amount
            self.finished_fee += transaction.fee
            self.delay += time - transaction.create_time

    def summary(self):
        """
        Summary report of transaction
        :return: A pandas data frame of transaction information.
        """
        stats = defaultdict(list)
        for transaction in self.history:
            stats['amount'].append(transaction.amount)
            stats['fee'].append(transaction.fee)
            stats['create_time'].append(transaction.create_time)
            stats['complete_time'].append(transaction.complete_time)
            stats['finished'].append(transaction.complete_time != None)
        stats = pd.DataFrame(stats)
        stats['pending_time'] = stats['complete_time'] - stats['create_time']
        return stats


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
        self.time_line['all_transaction_count'].append(
            pool.pending_transaction + pool.finished_transaction)
        self.time_line['pending_transaction_count'].append(
            pool.pending_transaction)
        self.time_line['finished_transaction_count'].append(
            pool.finished_transaction)
        self.time_line['all_transaction_amount'].append(
            pool.pending_amount + pool.finished_amount)
        self.time_line['pending_transaction_amount'].append(
            pool.pending_amount)
        self.time_line['finished_transaction_amount'].append(
            pool.finished_amount)
        self.time_line['all_transaction_fee'].append(
            pool.pending_fee + pool.finished_fee)
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
        data['average_delay'] = data['delay'] / \
            data['finished_transaction_count']
        data['average_transaction_amount'] = data[
            'all_transaction_amount'] / data['all_transaction_count']
        data['average_transaction_fee'] = data[
            'all_transaction_fee'] / data['all_transaction_count']
        data['average_pending_transaction_amount'] = data['pending_transaction_amount'] / data[
            'pending_transaction_count']
        data['average_pending_transaction_fee'] = data[
            'pending_transaction_fee'] / data['pending_transaction_count']
        data['average_finished_transaction_amount'] = data['finished_transaction_amount'] / data[
            'finished_transaction_count']
        data['average_finished_transaction_fee'] = data[
            'finished_transaction_fee'] / data['finished_transaction_count']
        data['average_block_size'] = data[
            'finished_transaction_count'] / data['block']
        data['average_block_amount'] = data[
            'finished_transaction_amount'] / data['block']
        data['average_block_fee'] = data[
            'finished_transaction_fee'] / data['block']
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
        heappush(self.event, (self.time + inter_arrival_time,
                              self.add_transaction_event))

    def generate_block_event(self):
        # Generate a block
        self.pool.generate_block(self.time)
        # Add next block generation event into the event pool
        heappush(self.event, (self.time + self.mu_generator(),
                              self.generate_block_event))

    def run(self, duration, warm_up=0, verbose=False):
        """
        Run simulation.
        :param duration: length of the simulation
        :param warm_up: warm up time
        :return: None
        """
        # Loop until there is no event
        counter = 0
        start_time = systime.time()
        while len(self.event) != 0:
            counter += 1
            if counter % 10000 == 0:
                print('Simulated %s events at time %s...' % (counter, time))
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
        self.transaction_stats = self.pool.summary()
        elapse = systime.time() - start_time
        print('Simulation finished')
        print('Simulated %s events, last %.1lf seconds, average speed is %.3lf events/s' %
              (counter, elapse, counter / elapse))

    def reset(self):
        """
        Reset the simulation system.
        :return: None
        """
        self.event = [(0, self.add_transaction_event),
                      (0, self.generate_block_event)]
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
            raise ValueError(
                'Only the following variables can be plotted. %s' % str(self.stats.columns))
        column = self.stats[response]
        sns.lineplot(x='time(minutes)', y=response, data=self.stats)

    def transaction_plot(self, x=None, y=None):
        """
        Plot transaction related variables.
        :param x, y: if only x is specified, then plot the distribution of x, otherwise plot the scatterplot
        :return: seaborn plot
        """
        if x not in self.transaction_stats.columns:
            raise ValueError('Only the following variables can be plotted. %s' % str(
                self.transaction_stats.columns))

        if y == None:
            sns.distplot(a=self.transaction_stats[x])
        elif y not in self.transaction_stats.columns:
            raise ValueError('Only the following variables can be plotted. %s' % str(
                self.transaction_stats.columns))
        else:
            sns.scatterplot(x=x, y=y, hue='finished',
                            data=self.transaction_stats)

if __name__ == '__main__':
    def transaction_generator():
        """
        Define a transaction generator, it should return 3 value, the transaction amount, fee 
        and an inter-arrival time. Notice that the unit for time is minute.
        """
        # Uniform transaction amount between 5, 25.
        amount = stats.uniform.rvs() * 20 + 5
        # Binomial transaction fee of 1% * amount or 2% * amount
        # fee = amount * (stats.binom.rvs(1, 0.5) / 100 + 0.01)
        fee = 0.1
        # Exponential inter-arrival time of 2 transaction / minute
        inter_arrival_time = stats.expon.rvs(scale=0.5)
        return amount, fee, inter_arrival_time

    def mu_generator():
        """
        Define a block event generator, it should return a random value of an inter-arrival time for block
        generation.
        """
        # Exponential inter-arrival time of 0.5 block / minute
        return stats.expon.rvs(scale=2)

    simulator = Simulator(transaction_generator, mu_generator, K=10)
    simulator.run(duration=5000, warm_up=1000)
