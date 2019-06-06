import numpy as np
import networkx as nx
from multiprocessing import Pool

from . import bandit
from . import agents


def runExperiment(bandit, agent_list, num_trials=100, num_threads=1):
    '''Run an experiment once the bandits and agents are finalized.'''

    use_pool = num_threads > 1
    for t in range(num_trials):
        # first agents pull arms
        pulled_arms = []
        for agent in agent_list:
            pulled_arms.append(agent.play())




if __name__ == '__main__':

    graphs = [bandit.generateRandomGraph(10, p=1) for _ in range(5)]
    num_edges = sum(len(list(g.edges)) for g in graphs)
    G = bandit.NetworkBandit(10, 5, eps=0.4)
    num_edges_new = sum(len(list(g.edges)) for g in G.graphs)
    # print(G.means.shape)
    print(G.verifyInit(), num_edges, num_edges_new, num_edges - num_edges_new)
