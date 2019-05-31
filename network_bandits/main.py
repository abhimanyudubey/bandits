import numpy as np
import networkx as nx
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


class NetworkBandit:

    @staticmethod
    def generateGraph(n=100, graph_type='er', **kwargs):
        ''' Generate a random graph with n nodes '''
        if graph_type == 'er':
            # generate erdos-renyi graph
            if 'p' in kwargs:
                return nx.gnp_random_graph(n, kwargs['p'])
            else:
                return nx.gnp_random_graph(n, 1)

    @staticmethod
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

    def __init__(
            self, num_agents, num_arms, graphs, mu_min=0,
            mu_max=1, dist='gaussian', eps=None, **kwargs):

        assert num_arms == len(graphs)
        assert mu_max >= mu_min
        assert eps >= 0 and eps <= mu_max - mu_min

        self.num_agents = num_agents
        self.num_arms = num_arms
        self.graphs = graphs

        clique_complete = zip(
            *[NetworkBandit.assignCliques(g) for g in self.graphs])
        self.cmap = clique_complete[0]
        self.cliques = clique_complete[1]

        if eps:
            # means are network bounded
            self.eps = eps
            self.bandit_type = 'bounded'

            temp_means = None
            for arm in range(self.num_arms):
                this_arm_means = {}
                for clique in self.cliques[arm]:
                    if clique[0] not in this_arm_means:
                        this_arm_means[clique[0]] =\
                            np.random.uniform(mu_min, mu_max)
                    base_mean = this_arm_means[clique[0]]
                    for node in clique[1:]:
                        if node not in this_arm_means:
                            this_arm_means[node] =\
                                base_mean + np.random.uniform(
                                    -self.eps, self.eps)
                means_transpose =\
                    [this_arm_means[x] for x in range(self.num_agents)]
                this_arm_means = np.expand_dims(np.asarray(means_transpose), 0)
                if temp_means is not None:
                    temp_means = np.concatenate(
                        (temp_means, this_arm_means))
                else:
                    temp_means = this_arm_means

            self.means = temp_means.transpose()

        else:
            # randomly initialize means within the range
            self.eps = None
            self.means = []
            self.bandit_type = 'random'
            for i in range(self.num_agents):
                this_means = []
                for j in range(self.num_arms):
                    this_means.append(np.random.uniform(mu_min, mu_max))
                self.means.append(this_means)

            self.means = np.asarray(self.means)

    def verifyInit(self):
        # verify if the assignments are all correct
        if self.eps is None:
            return self.means is not None
        else:
            for i, graph in enumerate(self.graphs):
                for edge in graph.edges:
                    src, dest = edge
                    print(src, dest)
                    edge_ok = np.abs(
                        self.means[src][i] - self.means[dest][i]) <= self.eps
                    if not edge_ok:
                        return False

            return True


class NetworkAgent:

    def __init__(self):
        return None


if __name__ == '__main__':

    graphs = [NetworkBandit.generateGraph(5, p=0.4) for _ in range(4)]
    G = NetworkBandit(5, 4, graphs, eps=0.1)

    print(G.verifyInit())
