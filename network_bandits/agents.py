import numpy as np


class NetworkAgent:

    def __init__(self, graphs, id):

        self.num_iters = 0
        self.arms = len(graphs)
        self.id = id
        self.neighbors = [list(g.neighbors(id) for g in graphs)]

        self.samples = np.zeros((self.arms, 1))
        self.best_arm = np.argmax(self.means)
        self.played_arm = None


class RandomAgent(NetworkAgent):

    def play(self):
        return np.random.choice(range(self.arms))

    def update(self, reward):
        pass


class BaseUCBAgent(NetworkAgent):

    def play(self):

        self.num_iters += 1

        if self.num_iters <= self.arms:
            self.played_arm = self.num_iters - 1

        else:
            if self.dist == 'bernoulli':
                conf_values = self.means +\
                    np.sqrt(np.log(self.num_iters)*0.5/self.samples)
            elif self.dist == 'gaussian':
                conf_values = self.means +\
                    np.sqrt(np.log(self.num_iters)*2/self.samples)
            # TODO: Implement alpha-stable version of UCB.

            self.played_arm = np.argmax(conf_values)

        return self.played_arm

    def update(self, reward):

        self.means[self.played_arm] =\
            (self.means[self.played_arm]*self.samples[self.played_arm] +
             reward)/(self.samples[self.played_arm]+1)
        self.samples[self.played_arm] += 1
