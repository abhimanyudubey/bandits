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


if __name__ == '__main__':

    G = NetworkBandit(30, 5, )
    cmap, cliques = assignCliques(G)

    print(len(cliques), cliques)
