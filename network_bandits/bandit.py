import numpy as np
import networkx as nx


def cms_alpha(_alpha, _beta, mu, sigma):
    ''' Generate a random sample from the alpha-stable distribution f
        using the CMS Algorithm, where f : S_alpha(beta, mu, sigma) '''

    b = np.arctan(_beta * np.tan(0.5 * np.pi * _alpha))*(1/_alpha)
    s = (1 + (_beta * np.tan(0.5 * np.pi * _alpha))**2)**(0.5/_alpha)

    v = np.random.uniform(-0.5*np.pi, 0.5*np.pi)
    w = np.random.exponential(1)

    if _alpha == 1:
        z = 2/np.pi * ((
            0.5*np.pi + _beta*v)*np.tan(v) - _beta *
            np.log((np.pi*0.5*w*np.cos(v))/(0.5*np.pi + _beta*v)))
    else:
        z = s * (np.sin(_alpha*(v + b))/(np.cos(v)**(1/_alpha))) *\
            (np.cos(v - _alpha*(v + b))/w)**((1-_alpha)/_alpha)

    return float(z*sigma + mu)


def generateRandomGraph(n=100, graph_type='er', **kwargs):
    ''' Generate a random graph with n nodes '''
    if graph_type == 'er':
        # generate erdos-renyi graph
        if 'p' in kwargs:
            return nx.gnp_random_graph(n, kwargs['p'])
        else:
            return nx.gnp_random_graph(n, 1)
    elif graph_type == 'complete':
        return nx.complete_graph(n)


def generateEpsGraph(n=100, eps=0.1, mu_min=0, mu_max=1, init_graph=None):
    '''Generate a graph by first initializing means and truncating edges'''

    base_graph = init_graph
    means = np.random.uniform(mu_min, mu_max, size=(n, 1))

    if not init_graph:
        base_graph = nx.complete_graph(n)
        for src in range(n):
            for dest in range(src+1, n):
                if np.abs(means[src] - means[dest]) > eps:
                    base_graph.remove_edge(src, dest)
    else:
        edge_set = list(base_graph.edges)[:]
        for edge in edge_set:
            src, dest = edge
            if np.abs(means[src] - means[dest]) > eps:
                base_graph.remove_edge(src, dest)

    return base_graph, means


def assignCliques(graph):
    ''' Assign to each node the largest clique it is a part of '''

    cliques = nx.find_cliques(graph)
    cliques_ptr = nx.find_cliques(graph)
    clique_map = {}
    clique_lengths = []

    for i, clique in enumerate(cliques):
        clique_lengths.append(len(clique))

        for node in clique:
            if node not in clique_map:
                clique_map[node] = i
            else:
                if clique_lengths[clique_map[node]] < len(clique):
                    clique_map[node] = i

    clique_counts = [0] * len(clique_lengths)
    for node, max_clique in clique_map.iteritems():
        clique_counts[max_clique] += 1

    shift_map = {}
    selected_index = 0
    selected_cliques = []
    for i, clique in enumerate(cliques_ptr):

        if clique_counts[i] > 0:
            shift_map[i] = selected_index
            selected_index += 1
            selected_cliques.append(clique)

    for node, max_clique in clique_map.iteritems():
        clique_map[node] = shift_map[max_clique]

    return clique_map, selected_cliques


class NetworkBandit:

    def __init__(
            self, num_agents, num_arms, base_graphs=None, mu_min=0,
            mu_max=1, dist='gaussian', eps=None, **kwargs):

        assert mu_max >= mu_min
        assert eps >= 0 and eps <= mu_max - mu_min

        self.num_agents = num_agents
        self.num_arms = num_arms
        self.graphs = []
        self.means = None
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.dist = dist

        if eps is not None:
            # There is a closeness constraint on the means.
            self.bandit_type = 'bounded'
            if type(eps) is not list:
                eps = [eps]*self.num_arms

            assert type(eps) is list
            self.eps = eps
            for k in range(self.num_arms):
                if base_graphs is not None:
                    this_graph, this_means = generateEpsGraph(
                            self.num_agents, self.eps[k], mu_min, mu_max,
                            base_graphs[k])
                else:
                    this_graph, this_means = generateEpsGraph(
                            self.num_agents, self.eps[k], mu_min, mu_max)
                self.graphs.append(this_graph)

                if self.means is None:
                    self.means = this_means
                else:
                    self.means = np.concatenate((self.means, this_means), 1)

        else:
            # There are no closeness constraints on the means.
            self.bandit_type = 'free'
            if base_graphs:
                assert num_arms == len(base_graphs)
                self.graphs = base_graphs

            else:
                self.graphs = []
                for k in range(self.num_arms):
                    if 'graph_type' not in kwargs:
                        kwargs['graph_type'] = 'complete'

                    self.graphs.append(generateRandomGraph(
                            self.num_agents, kwargs))

                self.means = np.random.uniform(
                    mu_min, mu_max,
                    (self.num_agents, self.num_arms))

        # setting variance parameters based on distribution
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

        self.average_rewards = np.zeros_like(self.means)
        self.pulls = np.zeros_like(self.means)
        self.iter = 0
        self.cliques_init = False

    def initCliques(self):
        ''' Initialize the clique assignment if required by the algorithm.'''
        if not self.cliques_init:
            print('Clique map not initialized, initializing now.')
            # TODO: Replace with logger
            cmap_l, cliq_l = [], []
            for graph in self.graphs:
                cmap, cliq = assignCliques(graph)
                cmap_l.append(cmap)
                cliq_l.append(cliq)

        self.clique_map = cmap_l
        self.cliques = cliq_l
        self.cliques_init = True

    def verifyInit(self):
        # verify if the assignments are all correct
        if self.eps is None:
            return self.means is not None
        else:
            for i, graph in enumerate(self.graphs):
                for edge in graph.edges:
                    src, dest = edge
                    edge_ok = np.abs(
                        self.means[src][i] -
                        self.means[dest][i]) <= self.eps[i]
                    if not edge_ok:
                        return False

            return True

    def getRewardSample(self, agent_id, arm):
        ''' Generate rewards for agent (randomly sampled)'''
        if self.dist is 'gaussian':
            reward = np.random.normal(
                self.means[agent_id][arm], self.sigma[agent_id][arm], 1)
        elif self.dist is 'bernoulli':
            reward = np.random.binomial(1, self.means[agent_id][arm])
        elif self.dist is 'stable':
            reward = cms_alpha(
                self.alpha[agent_id][arm], 0,
                self.means[agent_id][arm], self.sigma[agent_id][arm])

        self.average_rewards[agent_id][arm] = (
            self.average_rewards[agent_id][arm]*self.pulls[agent_id][arm] +
            reward)/(self.pulls[agent_id][arm]+1.0)
        self.pulls[agent_id][arm] += 1.0

        return reward

    def getRegret(self):
        '''Calculate the overall regret at any iteration of the game.'''

        obtained_rewards = np.sum(
            np.multiply(self.pulls, self.average_rewards))
        best_rewards = np.sum(np.max(self.means, axis=1)*self.iter)

        return best_rewards - obtained_rewards

    def getNeighborInformation(self, src, use_clique=False):
        '''Communicate information about neighbor rewards.'''

        if use_clique:
            if not self.cliques_init:
                # initialize the largest clique mapping first
                self.initCliques()

        averages, counts, ids = [], [], []
        for arm in range(self.num_arms):

            arm_averages, arm_counts, arm_ids = [], [], []

            if use_clique:
                for dest in self.cliques[arm][self.clique_map[arm][src]]:
                    arm_averages.append(self.average_rewards[dest][arm])
                    arm_counts.append(self.pulls[dest][arm])
                    arm_ids.append(dest)
            else:
                for dest in self.graphs[arm].neighbors(src):
                    arm_averages.append(self.average_rewards[dest][arm])
                    arm_counts.append(self.pulls[dest][arm])
                    arm_ids.append(dest)

            averages.append(arm_averages)
            counts.append(arm_counts)
            ids.append(arm_ids)

        return averages, counts, ids
