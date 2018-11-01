import numpy as np


class Environment:

    def __init__(self, n, dist='bernoulli', **kwargs):

        self.arms = n
        self.dist = dist

        if self.dist == 'bernoulli':

            if 'mean' in kwargs:
                assert len(kwargs['mean']) == n
                self.params = kwargs['mean']
            else:
                self.params = np.random.uniform(0, 1, n)

            self.best_arm = np.argmax(self.params)

    def iter(self, arm):

        if self.dist == 'bernoulli':

            assert arm < self.arms
            return np.random.binomial(1, self.params[arm], 1)
