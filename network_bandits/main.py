import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import normalize
import bandit
import agents
import random
import numpy as np
import sys


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

    regret_ucb1, regret_ucbM, regret_ucb2 = None, None, None
    rounds = 1
    min_agents, max_agents = 30, 100
    min_arms, max_arms = 20, 20
    num_iters = 10000
    eps = 1
    for round in range(rounds):

        num_agents = random.choice(range(min_agents, max_agents+1))
        num_arms = random.choice(range(min_arms, max_arms+1))

        this_round_regret = []
        graphs = [
            bandit.generateRandomGraph(num_agents, p=1)
            for _ in range(num_arms)]
        num_edges = sum(len(list(g.edges)) for g in graphs)
        G1 = bandit.NetworkBandit(
            num_agents,
            num_arms,
            graphs,
            eps=eps,
            mu_min=0,
            mu_max=1,
            sigma=5)
        G2 = copy.deepcopy(G1)
        G3 = copy.deepcopy(G1)
        G4 = copy.deepcopy(G1)
        G5 = copy.deepcopy(G1)

        random_agents = [
            agents.RandomAgent(graphs, x, 'gaussian') for x in
            range(num_agents)]
        ucb_agents = [
            agents.BaseUCBAgent(graphs, x, 'gaussian') for x in
            range(num_agents)]
        # maxucb_agents = [
        #     agents.MaxMeanUCBAgent(
        #         graphs, x, 'gaussian') for x in range(num_agents)]
        # ucb2_agents = [
        #     agents.UCBoverUCBAgent(
        #         graphs, x, 'gaussian', eps=eps) for x in range(num_agents)]
        deltats_agents = [
            agents.DeltaThompsonSamplingAgent(
                graphs,
                x,
                'gaussian',
                eps=eps,
                delta_ts=True)
            for x in range(num_agents)]

        ucb1_regret = runExperimentSingleCore(G2, ucb_agents, num_iters)
        # ucbM_regret = runExperimentSingleCore(G3, maxucb_agents, num_iters)
        # ucb2_regret = runExperimentSingleCore(G4, ucb2_agents, num_iters)
        deltats_regret = runExperimentSingleCore(G5, deltats_agents, num_iters)

        if regret_ucb1 is None:
            regret_ucb1 = np.expand_dims(np.array(ucb1_regret), 0)
            # regret_ucb2 = np.expand_dims(np.array(ucb2_regret), 0)
            # regret_ucbM = np.expand_dims(np.array(ucbM_regret), 0)
            regret_deltats = np.expand_dims(np.array(deltats_regret), 0)
        else:
            regret_ucb1 = np.concatenate(
                (regret_ucb1, np.expand_dims(np.array(ucb1_regret), 0)), 0)
            # regret_ucb2 = np.concatenate(
            #     (regret_ucb2, np.expand_dims(np.array(ucb2_regret), 0)), 0)
            # regret_ucbM = np.concatenate(
            #     (regret_ucbM, np.expand_dims(np.array(ucbM_regret), 0)), 0)
            regret_deltats = np.concatenate(
                (regret_deltats,
                    np.expand_dims(np.array(deltats_regret), 0)), 0)

        print('Completed round %d.' % (round+1))

    mean_ucb1 = np.mean(regret_ucb1, axis=0)
    # mean_ucb2 = np.mean(regret_ucb2, axis=0)
    # mean_ucbM = np.mean(regret_ucbM, axis=0)
    mean_dets = np.mean(regret_deltats, axis=0)

    std_ucb1 = np.std(regret_ucb1, axis=0)
    # std_ucbM = np.std(regret_ucbM, axis=0)
    # std_ucb2 = np.std(regret_ucb2, axis=0)
    std_dets = np.std(regret_deltats, axis=0)
    # plt.plot(range(num_iters), random_regret)

    plt.plot(range(num_iters), list(mean_ucb1), label='UCB-1', color='blue')
    plt.fill_between(
        range(num_iters), mean_ucb1-std_ucb1, mean_ucb1+std_ucb1,
        color='blue', alpha=0.2)
    # plt.plot(range(num_iters), mean_ucb2, label='UCB-2', color='green')
    # plt.fill_between(
    #     range(num_iters), mean_ucb2-std_ucb2, mean_ucb2+std_ucb2,
    #     color='green', alpha=0.2)
    # plt.plot(range(num_iters), mean_ucbM, label='UCB-M', color='red')
    # plt.fill_between(
    #     range(num_iters), mean_ucbM-std_ucbM, mean_ucbM+std_ucbM,
    #     color='red', alpha=0.2)
    plt.plot(range(num_iters), mean_dets, label='delta-TS', color='cyan')
    plt.fill_between(
        range(num_iters), mean_dets-std_dets, mean_dets+std_dets,
        color='red', alpha=0.2)
    plt.legend(loc='upper left')
    plt.savefig('result.png')
