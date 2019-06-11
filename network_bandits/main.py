import numpy as np
import networkx as nx
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import normalize

import bandit
import agents


def runExperimentSingleCore(bandit, agent_list, num_trials=100):
    '''Run experiment serially.'''

    regrets = []

    for t in range(num_trials):
        # agents will pull arms first
        played_arms = []
        for agent in agent_list:
            played_arms.append(agent.play())

        # get rewards
        rewards = []
        for id, arm in enumerate(played_arms):
            rewards.append(bandit.getRewardSample(id, arm))

        bandit.iter += 1

        round_regret = bandit.getRegret()

        # communicate external rewards
        for id, agent in enumerate(agent_list):
            avgs, counts, ids = bandit.getNeighborInformation(
                id, agent.uses_clique)
            agent.update(
                rewards[id], nbr_avgs=avgs, nbr_cnts=counts, nbr_ids=ids)

        regrets.append(round_regret)

    return regrets


def drawPullMap(output_file, bandits):

    plt.clf()

    for j, bx in enumerate(bandits):

        i = j + 1
        normalized_pulls = normalize(bx.pulls, axis=1, norm='l1')
        normalized_means = normalize(bx.means, norm='l1')

        plt.subplot(len(bandits), 2, 2*i-1)
        plt.imshow(normalized_pulls, cmap='hot', interpolation='nearest')
        plt.subplot(len(bandits), 2, 2*i)
        plt.imshow(normalized_means, cmap='hot', interpolation='nearest')

    plt.savefig(output_file)


if __name__ == '__main__':

    graphs = [bandit.generateRandomGraph(10, p=1) for _ in range(5)]
    num_edges = sum(len(list(g.edges)) for g in graphs)
    G1 = bandit.NetworkBandit(
        10, 5, graphs, eps=1, mu_min=0, mu_max=10, sigma=2.5)
    G2 = copy.deepcopy(G1)
    G3 = copy.deepcopy(G1)

    random_agents = [
        agents.RandomAgent(graphs, x, 'gaussian') for x in range(10)]
    ucb_agents = [
        agents.BaseUCBAgent(graphs, x, 'gaussian') for x in range(10)]
    maxucb_agents = [
        agents.MaxMeanUCBAgent(graphs, x, 'gaussian') for x in range(10)]

    random_regret = runExperimentSingleCore(G1, random_agents, 5000)
    ucb1_regret = runExperimentSingleCore(G2, ucb_agents, 5000)
    ucbM_regret = runExperimentSingleCore(G3, maxucb_agents, 5000)

    # plt.plot(range(5000), random_regret)
    plt.plot(range(5000), ucb1_regret)
    plt.plot(range(5000), ucbM_regret)
    plt.savefig('result.png')

    drawPullMap('result_pulls.png', [G1, G2])
