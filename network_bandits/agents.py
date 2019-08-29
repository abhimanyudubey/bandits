import numpy as np
import pymc3 as pm


def getConfidenceBound(dist_type, counts, sigma=1, **kwargs):

    if dist_type == 'bernoulli':
        return np.sqrt(np.log(counts)*0.5/counts)
    elif dist_type == 'gaussian':
        return np.sqrt(np.log(counts)*sigma*sigma*1.0/counts)


class NetworkAgent:

    def __init__(self, graphs, id, dist='gaussian', **kwargs):

        self.num_iters = 0
        self.arms = len(graphs)
        self.id = id
        self.neighbors = [list(g.neighbors(id) for g in graphs)]

        self.samples = np.zeros((self.arms, 1))
        self.played_arm = None
        self.means = [0.0]*self.arms
        self.dist = dist
        self.uses_clique = False

        self.nbr_avgs = None
        self.nbr_counts = None
        self.nbr_ids = None

        if self.dist == 'gaussian':
            if 'sigma' in kwargs:
                if type(kwargs['sigma']) in [int, float]:
                    self.sigma = np.full_like(self.means, kwargs['sigma'])
                elif type(kwargs['sigma']) is list:
                    assert len(kwargs['sigma']) == self.num_agents
                    assert len(kwargs['sigma'][0]) == self.num_arms
                    self.sigma = np.array(kwargs['sigma'])
                elif type(kwargs['sigma']) is np.ndarray:
                    assert kwargs['sigma'].shape ==\
                        [self.num_agents, self.num_arms]
                    self.sigma = kwargs['sigma']
            else:
                self.sigma = np.full_like(self.means, 1)
            self.sigma = np.expand_dims(self.sigma, axis=1)

        elif self.dist == 'stable':
            if 'alpha' in kwargs:
                if type(kwargs['alpha']) in [int, float]:
                    assert kwargs['alpha'] > 1
                    self.alpha = np.full_like(self.means, kwargs['alpha'])
                elif type(kwargs['alpha']) is list:
                    assert len(kwargs['alpha']) == self.num_agents
                    assert len(kwargs['alpha'][0]) == self.num_arms
                    assert all(np.greater(np.array(kwargs['alpha']), 1))
                    self.alpha = np.array(kwargs['alpha'])
                elif type(kwargs['alpha']) is np.ndarray:
                    assert kwargs['alpha'].shape ==\
                        [self.num_agents, self.num_arms]
                    assert all(np.greater(kwargs['alpha'], 1))
                    self.alpha = kwargs['alpha']
            else:
                self.alpha = np.full_like(self.means, 2)
            if 'sigma' in kwargs:
                if type(kwargs['sigma']) in [int, float]:
                    self.sigma = np.full_like(self.means, kwargs['sigma'])
                elif type(kwargs['sigma']) is list:
                    assert len(kwargs['sigma']) == self.num_agents
                    assert len(kwargs['sigma'][0]) == self.num_arms
                    self.sigma = np.array(kwargs['sigma'])
                elif type(kwargs['sigma']) is np.ndarray:
                    assert kwargs['sigma'].shape ==\
                        [self.num_agents, self.num_arms]
                    self.sigma = kwargs['sigma']
            else:
                self.sigma = np.full_like(self.means, 1)

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
                conf_values = np.array(self.means) +\
                    np.multiply(
                        self.sigma,
                        np.sqrt(np.log(self.num_iters)*0.5/self.samples))
            elif self.dist == 'gaussian':
                conf_values = np.array(self.means) +\
                    np.multiply(
                        self.sigma,
                        np.sqrt(np.log(self.num_iters)*2/self.samples))
            # TODO: Implement alpha-stable version of UCB.

            self.played_arm = np.argmax(conf_values)

        return self.played_arm

    def update(self, reward, **kwargs):

        self.means[self.played_arm] =\
            (self.means[self.played_arm]*self.samples[self.played_arm] +
             reward)/(self.samples[self.played_arm]+1)
        self.samples[self.played_arm] += 1


class MaxMeanUCBAgent(NetworkAgent):

    def play(self):

        if not self.uses_clique:
            self.uses_clique = True

        if self.num_iters <= 20:
            self.played_arm = (self.num_iters - 1) % self.arms

        elif self.nbr_ids is not None and self.num_iters > 200:

            base_confs, base_cnts = [], []
            # print(self.nbr_avgs, self.nbr_counts, self.nbr_ids)
            for arm, (avgx, cntx, idx) in enumerate(
                    zip(self.nbr_avgs, self.nbr_counts, self.nbr_ids)):
                # calculate the confidence values based on each arm
                sum_cnt = self.samples[arm]
                sum_rewards = self.means[arm]*sum_cnt
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
                    self.sigma*np.sqrt(np.log(self.num_iters)*0.5/base_cnts)
            elif self.dist == 'gaussian':
                conf_values = base_confs +\
                    self.sigma*np.sqrt(np.log(self.num_iters)*2/base_cnts)
            # TODO: Implement alpha-stable version of UCB.

            self.played_arm = np.argmax(conf_values)
        else:
            # no neighbors, use regular UCB
            if self.dist == 'bernoulli':
                conf_values = self.means +\
                    self.sigma*np.sqrt(np.log(self.num_iters)*0.5/self.samples)
            elif self.dist == 'gaussian':
                conf_values = self.means +\
                    self.sigma*np.sqrt(np.log(self.num_iters)*2/self.samples)
            # TODO: Implement alpha-stable version of UCB.

            self.played_arm = np.argmax(conf_values)

        self.num_iters += 1
        return self.played_arm

    def update(self, reward, **kwargs):

        self.means[self.played_arm] =\
            (self.means[self.played_arm]*self.samples[self.played_arm] +
             reward)/(self.samples[self.played_arm]+1)
        self.samples[self.played_arm] += 1
        if 'nbr_avgs' in kwargs:
            self.nbr_avgs = kwargs['nbr_avgs']
            self.nbr_counts = kwargs['nbr_cnts']
            self.nbr_ids = kwargs['nbr_ids']


class UCBoverUCBAgent(NetworkAgent):

    def __init__(self, graphs, id, dist, **kwargs):
        NetworkAgent.__init__(self, graphs, id, dist)
        if 'eps' in kwargs:
            self.eps = kwargs['eps']

    def play(self):

        if not self.uses_clique:
            self.uses_clique = True

        if self.num_iters <= self.arms:
            self.played_arm = (self.num_iters - 1) % self.arms

        elif self.nbr_ids is not None:

            conf_values = []
            for arm, (avgx, cntx, idx) in enumerate(
                    zip(self.nbr_avgs, self.nbr_counts, self.nbr_ids)):
                # calculate the confidence values based on each arm
                scale_factor = 1.0/(len(self.nbr_ids[arm])+1)
                base_conf = scale_factor * (
                    self.means[arm] +
                    getConfidenceBound(
                        self.dist, self.samples[arm], self.sigma[arm]))
                final_conf = 0.0
                final_conf += base_conf
                eps_factor = self.eps/len(self.nbr_ids[arm])
                for avgxx, cntxx in zip(avgx, cntx):
                    ucb_other = avgxx + getConfidenceBound(
                        self.dist, cntxx, self.sigma[arm])
                    if ucb_other*scale_factor + eps_factor < base_conf:
                        final_conf += ucb_other*scale_factor + eps_factor
                    else:
                        final_conf += base_conf
                conf_values.append(final_conf)

            self.played_arm = np.argmax(conf_values)

        else:
            # no neighbors, use regular UCB
            if self.dist == 'bernoulli':
                conf_values = self.means +\
                    np.sqrt(np.log(self.num_iters)*0.5/self.samples)
            elif self.dist == 'gaussian':
                conf_values = self.means +\
                    np.sqrt(np.log(self.num_iters)*2/self.samples)
            # TODO: Implement alpha-stable version of UCB.

            self.played_arm = np.argmax(conf_values)

        self.num_iters += 1
        return self.played_arm

    def update(self, reward, **kwargs):

        self.means[self.played_arm] =\
            (self.means[self.played_arm]*self.samples[self.played_arm] +
             reward)/(self.samples[self.played_arm]+1)
        self.samples[self.played_arm] += 1
        if 'nbr_avgs' in kwargs:
            self.nbr_avgs = kwargs['nbr_avgs']
            self.nbr_counts = kwargs['nbr_cnts']
            self.nbr_ids = kwargs['nbr_ids']


class WeightedExpUCBAgent(NetworkAgent):

    def __init__(self, graphs, id, dist, **kwargs):
        NetworkAgent.__init__(self, graphs, id, dist)
        if 'eps' in kwargs:
            self.eps = kwargs['eps']

    def play(self):

        if not self.uses_clique:
            self.uses_clique = True

        if self.num_iters <= self.arms:
            self.played_arm = (self.num_iters - 1) % self.arms

        elif self.nbr_ids is not None:

            conf_values = []
            for arm, (avgx, cntx, idx) in enumerate(
                    zip(self.nbr_avgs, self.nbr_counts, self.nbr_ids)):
                # calculate the confidence values based on each arm
                w_arm = []
                for avgxx, cntxx in zip(avgx, cntx):
                    w_i = np.exp(-np.abs(cntxx - self.samples[arm]))
                sigma_f = self.sigma[arm]

                scale_factor = 1.0/(len(self.nbr_ids[arm])+1)
                base_conf = scale_factor * (
                    self.means[arm] +
                    getConfidenceBound(
                        self.dist, self.samples[arm], self.sigma[arm]))
                final_conf = 0.0
                final_conf += base_conf
                eps_factor = self.eps/len(self.nbr_ids[arm])
                for avgxx, cntxx in zip(avgx, cntx):
                    ucb_other = avgxx + getConfidenceBound(
                        self.dist, cntxx, self.sigma[arm])
                    if ucb_other*scale_factor + eps_factor < base_conf:
                        final_conf += ucb_other*scale_factor + eps_factor
                    else:
                        final_conf += base_conf
                conf_values.append(final_conf)

            self.played_arm = np.argmax(conf_values)

        else:
            # no neighbors, use regular UCB
            if self.dist == 'bernoulli':
                conf_values = self.means +\
                    np.sqrt(np.log(self.num_iters)*0.5/self.samples)
            elif self.dist == 'gaussian':
                conf_values = self.means +\
                    np.sqrt(np.log(self.num_iters)*2/self.samples)
            # TODO: Implement alpha-stable version of UCB.

            self.played_arm = np.argmax(conf_values)

        self.num_iters += 1
        return self.played_arm

    def update(self, reward, **kwargs):

        self.means[self.played_arm] =\
            (self.means[self.played_arm]*self.samples[self.played_arm] +
             reward)/(self.samples[self.played_arm]+1)
        self.samples[self.played_arm] += 1
        if 'nbr_avgs' in kwargs:
            self.nbr_avgs = kwargs['nbr_avgs']
            self.nbr_counts = kwargs['nbr_cnts']
            self.nbr_ids = kwargs['nbr_ids']


class DeltaThompsonSamplingAgent(NetworkAgent):

    def __init__(self, graphs, id, dist, **kwargs):
        NetworkAgent.__init__(self, graphs, id, dist)
        if 'eps' in kwargs:
            self.eps = kwargs['eps']

        dist_type = kwargs['dist']
        if dist_type == 'gaussian':
            if 'mu' in kwargs:
                self.prior_mean = kwargs['mu']
            else:
                self.prior_mean = 0.0

            if 'sigma' in kwargs:
                self.prior_sigma = kwargs['sigma']
            else:
                self.prior_sigma = 1.0

        # create pymc3 model
        self.prior_model = pm.Model()
        with self.prior_model:
            # just modeling the prior of the mean
            self.priors_mean = [
                pm.Normal('mu_arm_%d' % i, self.prior_mean, self.prior_sigma)
                for i in range(0, self.arms)]
            self.priors_delta = [[
                pm.Normal('delta_%d_arm_%d' % (i, j), 0, self.eps) for
                j in range(0, self.arms)] for i in
                range(0, len(self.neighbors))]

    def play(self):

        self.num_iters += 1

        with self.prior_model:
            if self.dist == 'gaussian':
                posterior_means = [pm.sample()]
