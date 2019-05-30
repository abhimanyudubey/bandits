import numpy as np
import networkx as nx


def generateGraph(n=100, e=100, graph_type='er'):
    ''' Generate a random graph with n nodes and e edges under a graph_type '''
    if graph_type == 'er':
        # generate erdos-renyi graph
        return nx.gnm_random_graph(n, e)


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


def initParamsForGraph(Gs, K, eps=None):
    ''' Initialize the mean rewards for graph G in Gs and K arms '''

    assert len(Gs) == K


if __name__ == '__main__':

    G = generateGraph(20, 50)
    cmap, cliques = assignCliques(G)

    print(cmap)
