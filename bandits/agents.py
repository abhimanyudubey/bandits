import random
import numpy as np


class RandomAgent:

    def __init__(self, n_arms):
        self.arms = n_arms

    def play(self):
        return random.choice(range(self.arms))

    def update(self, reward):
        pass


class UCB1Agent:

    def __init__(self, n_arms, dist='bernoulli', **kwargs):

        self.arms = n_arms
        self.num_iters = 0
        self.dist = dist

        if self.dist == 'bernoulli':

            if 'mean' in kwargs:
                assert len(kwargs['mean']) == n_arms
                self.means = kwargs['mean']

            else:
                self.means = np.zeros((n_arms, 1))

            self.samples = np.zeros((n_arms, 1))
            self.best_arm = np.argmax(self.means)

    def play(self):

        self.num_iters += 1

        if self.num_iters <= self.arms:
            self.played_arm = self.num_iters - 1

        else:
            if self.dist == 'bernoulli':
                conf_values = self.means +\
                    np.sqrt(np.log(self.num_iters)/self.samples)

            self.played_arm = np.argmax(conf_values)

        return self.played_arm

    def update(self, reward):

        self.means[self.played_arm] =\
            (self.means[self.played_arm]*self.samples[self.played_arm] +
             reward)/(self.samples[self.played_arm]+1)
        self.samples[self.played_arm] += 1


class ThompsonSamplingAgent:

    def __init__(self, n_arms, dist='bernoulli', pdist='beta', **kwargs):

        self.arms = n_arms
        self.num_iters = 0
        self.dist = dist
        self.pdist = pdist

        if self.dist == 'bernoulli':

            if 'mean' in kwargs:
                assert len(kwargs['mean']) == n_arms
                self.means = kwargs['mean']

            else:
                self.means = np.zeros((n_arms, 1))

            self.samples = np.zeros((n_arms, 1))
            self.best_arm = np.argmax(self.means)

    def play(self):

        self.num_iters += 1

        if self.num_iters <= self.arms:
            self.played_arm = self.num_iters - 1

        else:
            if self.dist == 'bernoulli':
                conf_values = self.means +\
                    np.sqrt(np.log(self.num_iters)/self.samples)

            self.played_arm = np.argmax(conf_values)

        return self.played_arm
