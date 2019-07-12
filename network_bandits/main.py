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

    plt.subplot(1, len(bandits)+1, 1)
    plt.title('Means')
    plt.xlabel('Arms = %d' % bandits[0].num_arms)
    plt.ylabel('Agents = %d' % bandits[0].num_agents)
    normalized_means = normalize(bandits[0].means, norm='l1')
    plt.imshow(normalized_means, cmap='hot', interpolation='nearest')

    for j, bx in enumerate(bandits):

        i = j + 1
        normalized_pulls = normalize(bx.pulls, axis=1, norm='l1')
        normalized_means = normalize(bx.means, norm='l1')

        plt.subplot(1, len(bandits)+1, i+1)
        plt.axis('off')
        plt.imshow(normalized_pulls, cmap='hot', interpolation='nearest')
        plt.title('Normalized Pulls')
        plt.xlabel('Arms = %d' % bandits[0].num_arms)
        plt.ylabel('Agents = %d' % bandits[0].num_agents)

    plt.savefig(output_file, bbox_inches='tight', dpi=600)


if __name__ == '__main__':

    num_agents = 100
    num_arms = 20
    num_iters = 2000
    graphs = [
        bandit.generateRandomGraph(num_agents, p=1) for _ in range(num_arms)]
    num_edges = sum(len(list(g.edges)) for g in graphs)
    G1 = bandit.NetworkBandit(
        num_agents, num_arms, graphs, eps=0.5, mu_min=0, mu_max=1, sigma=10)
    G2 = copy.deepcopy(G1)
    G3 = copy.deepcopy(G1)
    G4 = copy.deepcopy(G1)

    random_agents = [
        agents.RandomAgent(graphs, x, 'gaussian') for x in range(num_agents)]
    ucb_agents = [
        agents.BaseUCBAgent(graphs, x, 'gaussian') for x in range(num_agents)]
    maxucb_agents = [
        agents.MaxMeanUCBAgent(
            graphs, x, 'gaussian') for x in range(num_agents)]
    ucb2_agents = [
        agents.UCBoverUCBAgent(
            graphs, x, 'gaussian', eps=0.2) for x in range(num_agents)]

    # random_regret = runExperimentSingleCore(G1, random_agents, num_iters)
    ucb1_regret = runExperimentSingleCore(G2, ucb_agents, num_iters)
    ucbM_regret = runExperimentSingleCore(G3, maxucb_agents, num_iters)
    ucb2_regret = runExperimentSingleCore(G4, ucb2_agents, num_iters)

    # plt.plot(range(num_iters), random_regret)
    plt.plot(range(num_iters), ucb1_regret)
    plt.plot(range(num_iters), ucbM_regret)
    plt.plot(range(num_iters), ucb2_regret)
    plt.savefig('result.png')

    drawPullMap('result_pulls.png', [G2, G3, G4])
