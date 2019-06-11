import numpy as np


class NetworkAgent:

    def __init__(self, graphs, id, dist='gaussian'):

        self.num_iters = 0
        self.arms = len(graphs)
        self.id = id
        self.neighbors = [list(g.neighbors(id) for g in graphs)]

        self.samples = np.zeros((self.arms, 1))
        self.played_arm = None
        self.means = [0.0]*self.arms
        self.dist = dist
        self.uses_clique = False

    def update(self, reward, **kwargs):
        pass


class RandomAgent(NetworkAgent):

    def play(self):
        return np.random.choice(range(self.arms))

    def update(self, reward, **kwargs):
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

    def update(self, reward, **kwargs):

        self.means[self.played_arm] =\
            (self.means[self.played_arm]*self.samples[self.played_arm] +
             reward)/(self.samples[self.played_arm]+1)
        self.samples[self.played_arm] += 1


class MaxMeanUCBAgent(NetworkAgent):

    def __init__(self, graphs, id, dist='gaussian'):

        super(MaxMeanUCBAgent, self).__init__(graphs, id, dist)
        self.uses_clique = True
        self.nbr_avgs = None
        self.nbr_counts = None
        self.nbr_ids = None

    def play(self):

        if self.num_iters <= self.arms:
            self.played_arm = self.num_iters - 1

        else:

            base_confs, base_cnts = [], []
            for arm, avgx, cntx, idx in enumerate(
                    zip(self.nbr_avgs, self.nbr_counts, self.nbr_ids)):
                # calculate the confidence values based on each arm
                sum_cnt = 0
                sum_rewards = 0.0
                for avgxx, cntxx in zip(avgx, cntx):
                    if avgxx < self.means[arm]:
                        sum_cnt += cntxx
                        sum_rewards += avgxx*cntxx
                sum_rewards /= sum_cnt
                base_confs.append(sum_rewards)
                base_cnts.append(sum_cnt)

            base_confs = np.array(base_confs)
            base_cnts = np.array(base_cnts)
            if self.dist == 'bernoulli':
                conf_values = base_confs +\
                    np.sqrt(np.log(self.num_iters)*0.5/base_cnts)
            elif self.dist == 'gaussian':
                conf_values = base_confs +\
                    np.sqrt(np.log(self.num_iters)*2/base_cnts)
            # TODO: Implement alpha-stable version of UCB.

            self.played_arm = np.argmax(conf_values)

        self.num_iters += 1
        return self.played_arm

    def update(self, reward, **kwargs):

        self.means[self.played_arm] =\
            (self.means[self.played_arm]*self.samples[self.played_arm] +
             reward)/(self.samples[self.played_arm]+1)
        self.samples[self.played_arm] += 1
        self.nbr_avgs = kwargs['nbr_avgs']
        self.nbr_counts = kwargs['nbr_cnts']
        self.nbr_ids = kwargs['nbr_ids']
