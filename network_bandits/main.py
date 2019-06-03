import numpy as np
import networkx as nx
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
        if not init_graph:
            base_graph = nx.complete_graph(n)

        means = np.random.uniform(mu_min, mu_max, size=(n, 1))

        for src in range(n):
            for dest in range(src+1, n):
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

        if eps:
            #There is a closeness constraint on the means.
            self.bandit_type = 'bounded'
            if type(eps) is not list:
                eps = [eps]*self.num_arms

            assert type(eps) is list
            self.eps = eps

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



        if eps:

            temp_means = None
            for arm in range(self.num_arms):
                this_arm_means = {}
                print(self.graphs[arm].edges)
                edge_list = list(nx.bfs_edges(self.graphs[arm], 0))

                def bfs_assign(edge_list, start_idx, depth, key_store, base_value):
                    neighbors = [x[1] for x in edge_list if x[0] == start_idx]

                    if start_idx not in key_store:
                        key_store[start_idx] = base_value + np.random.uniform(
                            -self.eps[arm], self.eps[arm])

                    for neighbor in neighbors:
                        bfs_assign(edge_list, neighbor, depth+1, key_store, key_store[start_idx])

                bfs_assign(edge_list, 0, 0, this_arm_means, np.random.uniform(mu_min, mu_max))
                print(this_arm_means)
                # for node in range(self.num_agents):
                #     if node not in this_arm_means:
                #         this_arm_means[node] =\
                #             np.random.uniform(mu_min, mu_max)
                #     for dest_node in self.graphs[arm].neighbors(node):
                #         if dest_node not in this_arm_means:
                #             this_arm_means[dest_node] =\
                #                 this_arm_means[node] + np.random.uniform(
                #                     -self.eps[arm]*0.5, self.eps[arm]*0.5)
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
                    print(i, src, dest)
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

    graphs = [NetworkBandit.generateGraph(10, p=0.5) for _ in range(1)]
    G = NetworkBandit(10, 1, graphs, eps=0.1)

    print(G.verifyInit())
