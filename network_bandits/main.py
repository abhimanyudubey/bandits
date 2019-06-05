import numpy as np
import networkx as nx


class NetworkBandit:

    @staticmethod
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

    @staticmethod
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

        if eps:
            # There is a closeness constraint on the means.
            self.bandit_type = 'bounded'
            if type(eps) is not list:
                eps = [eps]*self.num_arms

            assert type(eps) is list
            self.eps = eps
            for k in range(self.num_arms):
                if base_graphs is not None:
                    this_graph, this_means =\
                        NetworkBandit.generateEpsGraph(
                            self.num_agents, self.eps[k], mu_min, mu_max,
                            base_graphs[k])
                else:
                    this_graph, this_means =\
                        NetworkBandit.generateEpsGraph(
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

                    self.graphs.append(
                        NetworkBandit.generateRandomGraph(
                            self.num_agents, kwargs))

                self.means = np.random.uniform(
                    mu_min, mu_max,
                    (self.num_agents, self.num_arms))

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


class NetworkAgent:

    def __init__(self):
        return None


if __name__ == '__main__':

    graphs = [NetworkBandit.generateRandomGraph(10, p=1) for _ in range(5)]
    num_edges = sum(len(list(g.edges)) for g in graphs)
    G = NetworkBandit(10, 5, eps=0.4)
    num_edges_new = sum(len(list(g.edges)) for g in G.graphs)
    # print(G.means.shape)
    print(G.verifyInit(), num_edges, num_edges_new, num_edges - num_edges_new)
