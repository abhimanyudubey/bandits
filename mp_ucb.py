import numpy as np
import networkx as nx
import dwave_networkx as dnx
import dimod
import math
from scipy.special import gamma
import argparse
import multiprocessing
import cPickle as pkl
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


USE_RANDOM_SAMPLER = False
if USE_RANDOM_SAMPLER:
    sampler = dimod.RandomSampler()
else:
    sampler = dimod.SimulatedAnnealingSampler()
dnx.set_default_sampler(sampler)


def cms_alpha(_alpha, _beta, mu, sigma):

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


def generate_connected_graph(n, graph_type='er'):
    # we generate a connected graph by first generating a random path graph and
    # ensuring it is connected
    base_path_graph = nx.path_graph(n)

    if graph_type == 'er':
        main_graph = nx.gnp_random_graph(n, 0.05)

    for edge in base_path_graph:
        if not main_graph.has_edge(*edge):
            main_graph.add_edge(*edge)

    return main_graph


def trimmed_mean(samples, u, eps, delta):
    # return the trimmed mean as defined in https://arxiv.org/pdf/1209.1727.pdf
    trimmed_mean = 0.0
    bound = (u*math.log(1.0/delta))**(1.0/(1+eps))
    for i, sample in enumerate(samples):
        if abs(sample) <= bound*((i+1)**(1.0/(1+eps))):
            bound += sample
    return trimmed_mean/(len(samples))


def median_of_means(samples, delta):
    # return the MoM as defined in https://arxiv.org/pdf/1209.1727.pdf
    n = len(samples)*1.0
    k = math.floor(max(8*math.log(math.e**0.125/delta), n/2))
    N = math.floor(n/k)
    means = []
    for i in range(k):
        mean_i = np.mean(samples[i*N:max(n, (i+1)*N)])
        means.append(mean_i)

    return np.median(means)


def get_ucb(
        samples, estimator_type, delta, p, alpha, sigma=1.0,
        is_heavy_tailed=True, max_mu=1.0):
    # get UCB given the robust mean estimator (empirical, trimmed or MoM)
    n = len(samples)

    if is_heavy_tailed:
        # if alpha-stable, calculate all the moments beforehand
        v = (
            (2**(p+1)*gamma((p+1.0)/2)*gamma(-p/alpha)*sigma**(p/alpha)) /
            (alpha*math.sqrt(math.pi)*gamma(-p*0.5)))
        u = ((max_mu*gamma((1-p)/alpha)+sigma*alpha*gamma(1-p/alpha))
            * (sigma**p)*(p-1))/(sigma*alpha*math.sin(math.pi/2*(p-1))
            * gamma(2-p))

        if estimator_type == 'empirical':
            return np.mean(samples) + ((3.0*v)/(delta*n**(p-1)))**(1/p)
        elif estimator_type == 'trimmed':
            return trimmed_mean(samples, u, p-1, delta) +\
                4*u**(1/p)*((math.log(1.0/delta)/n)**((p-1)/p))
        else:
            z = (12*v)**(1/p)*((16.0/n*math.log(
                (math.e**0.125)/delta)) ** (1-1.0/p))
            return median_of_means(samples, delta) + z

    else:
        return np.mean(samples) + sigma*math.sqrt(2.0*math.log(1.0/delta)/n)


def select_arm(
        samples, t, estimator_type, p, alpha, sigma=1.0,
        is_heavy_tailed=True, max_mu=1.0):
        # select arm following basic robust UCB strategy
    num_arms = len(samples)
    delta = (t*1.0)**(-2)

    if t <= num_arms:
        return t-1
    else:
        chosen_arm, max_ucb = 0, -np.inf
        for arm in range(num_arms):
            ucb_arm = get_ucb(
                samples[arm], estimator_type, delta, p, alpha,
                sigma, is_heavy_tailed, max_mu)
            if ucb_arm > max_ucb:
                chosen_arm = arm
                max_ucb = ucb_arm
        return chosen_arm


class Agent:

    def __init__(
            self, id, arms, neighbors, g, alpha, p, sigma, estimator_type,
            is_heavy_tailed=True, is_leader=True, leader_id=None, max_mu=1.0):

        self.is_leader = is_leader
        self.max_mu = max_mu
        self.neighbors = neighbors
        self.g = g
        self.alpha = alpha
        self.p = p
        self.sigma = sigma
        self.arms = arms
        self.is_heavy_tailed = is_heavy_tailed

        self.samples = []
        self.messages = []
        self.leader_last_action = None

        for _ in range(self.arms):
            self.samples.append([])

        self.id = id
        self.t = 0
        self.leader_id = leader_id
        self.pulls = []
        for _ in range(self.arms):
            self.pulls.append(0)
        self.self_rewards = []
        self.self_sum = 0.0
        self.estimator_type = estimator_type
        self.message_hash = {}

    def pull(self, env):

        if self.is_leader or self.leader_last_action is None:
            arm = select_arm(
                self.samples, self.t, self.estimator_type, self.p,
                self.alpha, self.sigma, self.is_heavy_tailed, self.max_mu)
        else:
            arm = self.leader_last_action

        return arm, env.reward(arm)

    def read_messages(self):

        saved_messages = []
        for message in self.messages:
            # each message is <id, life, arm, reward>
            if hash(tuple(message)) not in self.message_hash:

                if message[0] != self.id:
                    # only read messages that aren't from the agent itself
                    if self.is_leader:
                        # leaders save all messages
                        message_arm = message[2]
                        message_reward = message[3]
                        self.samples[message_arm].append(message_reward)
                    elif message[0] == self.leader_id:
                        self.leader_last_action = message[2]

                    self.message_hash[hash(tuple(message))] = 1

                # discard old messages
                if message[1] > 0:
                    new_message = message[:]
                    new_message[1] -= 1
                    saved_messages.append(message)

        self.messages = saved_messages

    def round(self, env):
        # round in the bandit environment
        self.t += 1
        self.read_messages()
        arm, reward = self.pull(env)
        self.samples[arm].append(reward)
        self.pulls[arm] += 1
        self.self_rewards.append(reward)
        self.self_sum += reward
        self.messages.append([self.id, self.g, arm, reward])


class Environment:

    def __init__(
            self, K, alpha, sigma=1.0, is_heavy_tailed=True, mu_min=0.0,
            mu_max=1.0):

        self.means = []
        self.alpha = alpha
        self.sigma = sigma
        self.K = K
        self.mu_max = mu_max
        self.mu_min = mu_min
        self.is_heavy_tailed = is_heavy_tailed

        self.p = alpha - 0.05
        for _ in range(K):
            self.means.append(np.random.uniform(mu_min, mu_max))

    def reward(self, arm):

        if self.is_heavy_tailed:
            return cms_alpha(self.alpha, 0, self.means[arm], self.sigma)
        else:
            return np.random.normal(self.means[arm], self.sigma)


class Manager:

    def __init__(
            self, N, g, alg_type=None, estimator_type='trimmed',
            graph=None, graph_params=None):
        # generate graph first
        if graph is None:
            graph = generate_connected_graph(N, 'er')

        power_graph = nx.power(graph, g)
        leaders, leader_assignment = [], []
        if alg_type == 'decentralized':
            # decentralized, everyone is a leader
            for i in range(N):
                leaders.append(i)
                leader_assignment.append(i)

        elif alg_type == 'ftl':
            # follow the leader, use weights to compute leaders
            weight_type = graph_params['node_weight']
            choice_type = graph_params['choice_type']

            weights = []
            if weight_type == 'degree':
                for agent in range(N):
                    agent_weight = 1.0*len([
                        x for x in power_graph.neighbors(agent)])
                    weights.append(agent_weight)
            elif weight_type == 'distance':
                for agent in range(N):
                    dist_sum = 0.0
                    nbrhood = [
                        x for x in power_graph.neighbors(agent)]
                    for nbr in nbrhood:
                        d_nbr = len(nx.shortest_path(
                            graph, source=agent, target=nbr))
                        dist_sum += 1.0/d_nbr
                    weights.append(dist_sum)
            else:
                bc = nx.betweenness_centrality(graph)
                for w in bc.keys():
                    weights.append(1.0*bc[w])

            weight_dict = dict(zip(range(N), weights))
            nx.set_node_attributes(graph, weight_dict, 'weights')
            # now to find dominating set
            leaders = dnx.maximum_weighted_independent_set(
                power_graph, 'weights')
            leader_assignment = []
            for agent in range(N):
                if agent in leaders:
                    leader_assignment.append(agent)
                else:
                    connected_leaders = [
                        x for x in leaders if power_graph.has_edge(x, agent)]
                    if choice_type == 'distance':
                        distances = [
                            len(nx.shortest_path(graph, x, agent))
                            for x in connected_leaders]
                        chosen_leader = connected_leaders[np.argmin(distances)]
                    else:
                        degrees = [
                            len([y for y in power_graph.neighbors(x)])
                            for x in connected_leaders]
                        chosen_leader = connected_leaders[np.argmax(degrees)]
                    leader_assignment.append(chosen_leader)

        # creating agents now
        self.n = N
        self.agents = []
        self.graph = graph
        self.power_graph = power_graph
        self.g = g
        self.alg_type = alg_type
        self.estimator_type = estimator_type
        self.graph_params = graph_params
        self.leaders = leaders
        self.leader_assignment = leader_assignment
        self.rounds = 0

    def create_agents(self, env):

        for agent in range(self.n):

            agent_nbrs = [x for x in self.graph.neighbors(agent)]
            agent_leader = self.leader_assignment[agent]
            agent_is_leader = agent == agent_leader

            self.agents.append(
                Agent(
                    agent, env.K, agent_nbrs, self.g, env.alpha, env.p,
                    env.sigma, self.estimator_type, env.is_heavy_tailed,
                    agent_is_leader, agent_leader, env.mu_max))

    def round(self, env):

        self.rounds += 1
        tmp_messages = []
        total_messages = 0
        for _ in range(self.n):
            tmp_messages.append([])

        for agent in self.agents:
            for nbr in agent.neighbors:
                tmp_messages[nbr].extend(agent.messages)

        for i, agent in enumerate(self.agents):
            agent.messages = tmp_messages[i]
            total_messages += len(tmp_messages[i])

        for agent in self.agents:
            agent.round(env)

        return self.get_regret(env)

    def draw_graph(self, pos=None):
        colormap = [[int(x in self.leaders), 0, 0] for x in range(self.n)]
        print(colormap)
        nx.draw(self.graph, pos=pos, node_color=colormap, cmap=plt.cm.Blues)

    def draw_power_graph(self, pos=None):
        colormap = [int(x in self.leaders) for x in range(self.n)]
        nx.draw(
            self.power_graph, pos=pos, node_color=colormap, cmap=plt.cm.Blues)

    def get_regret(self, env):

        top_mu = np.max(env.means)
        sum_rewards = self.rounds*top_mu\
            - 1.0/self.n*sum(x.self_sum for x in self.agents)
        max_regret = self.rounds*top_mu - min(x.self_sum for x in self.agents)
        min_regret = self.rounds*top_mu - max(x.self_sum for x in self.agents)

        return sum_rewards, max_regret, min_regret

    def run(self, T, env):

        regrets = {}
        regrets['min_regrets'] = []
        regrets['max_regrets'] = []
        regrets['avg_regrets'] = []

        for t in range(T):
            avg_r, max_r, min_r = self.round(env)
            regrets['min_regrets'].append(min_r)
            regrets['max_regrets'].append(max_r)
            regrets['avg_regrets'].append(avg_r)

        return regrets


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--experiment_type', type=int)
    parser.add_argument(
        '-n', '--num_agents', type=int, required=False, default=50)
    parser.add_argument(
        '-t', '--graph-type', type=str, required=False, default='er')
    parser.add_argument(
        '-g', '--gamma', type=int, required=False, default=None)

    args = parser.parse_args()
    if args.g is None:
        args.g == math.ceil(math.sqrt(args.n))
    exp_type = args.e

    threads = []
    thread_manager = multiprocessing.Manager()
    regret_dict_base = thread_manager.list()

    def exec_thread(
            n, graph, graph_params, env, T, g, k, gt, regret_dict_base):
        manager = Manager(
            n, g, alg_type, graph=graph, graph_params=graph_params)

        manager.create_agents(env)
        regrets = manager.run(T, env)

        regret_dict_base.append((k, g, n, gt, regrets))

        print('Done', round, g, k)

    if exp_type == 1:
        # compare regret over different connectivity
        n = args.n
        num_rounds = 10
        T = 1000
        K = 20

        env = Environment(K, 2, is_heavy_tailed=False)

        graph_params = {}
        graph_params['choice_type'] = 'degree'

        for round in range(num_rounds):
            graph = generate_connected_graph(n, args.t)
            diam = nx.diameter(graph)

            for g in range(1, diam+1):

                for k in [
                        'ftl_degree',
                        'ftl_distance',
                        'ftl_bc',
                        'decentralized']:

                    if k == 'ftl_degree':
                        alg_type = 'ftl'
                        graph_params['node_weight'] = 'degree'
                    elif k == 'ftl_distance':
                        alg_type = 'ftl'
                        graph_params['node_weight'] = 'distance'
                    elif k == 'ftl_bc':
                        alg_type = 'ftl'
                        graph_params['node_weight'] = 'bc'
                    else:
                        alg_type = 'decentralized'

                    process = multiprocessing.Process(
                        target=exec_thread,
                        args=[
                            n, graph, graph_params, env, T, g, k, args.t,
                            regret_dict_base])
                    process.daemon = True
                    process.start()
                    threads.append(process)

        for process in threads:
            process.join()

        regret_dict = {}
        for k in [
                'ftl_degree',
                'ftl_distance',
                'ftl_bc',
                'decentralized']:
            regret_dict[k] = {}

        for elem in regret_dict_base:
            k, g, nn, gt, regrets_round = elem
            if g not in regret_dict[k]:
                regret_dict[k][g] = regrets_round

        dt = '{date:%Y-%m-%d_%H:%M:%S}.pkl'.format(
            date=datetime.datetime.now())
        output_file = 'exp1_%d_%s_%s' % (args.n, args.t, dt)
        pkl.dump(
            (regret_dict, args.t, n), open(output_file, 'wb'),
            protocol=pkl.HIGHEST_PROTOCOL)

    if exp_type == 2:
        n_range = [1, 5, 10, 20, 40, 80, 160, 320]
        num_rounds = 10
        T = 1000
        K = 20

        env = Environment(K, 2, is_heavy_tailed=False)

        graph_params = {}
        graph_params['choice_type'] = 'degree'

        for n in n_range:
            graph = generate_connected_graph(n, args.t)
            diam = nx.diam(graph)
            g = math.ceil(math.sqrt(n))

            for k in ['ftl_degree', 'ftl_distance', 'ftl_bc', 'decentralized']:

                if k == 'ftl_degree':
                    alg_type = 'ftl'
                    graph_params['node_weight'] = 'degree'
                elif k == 'ftl_distance':
                    alg_type = 'ftl'
                    graph_params['node_weight'] = 'distance'
                elif k == 'ftl_bc':
                    alg_type = 'ftl'
                    graph_params['node_weight'] = 'bc'
                else:
                    alg_type = 'decentralized'

                process = multiprocessing.Process(
                    target=exec_thread,
                    args=[
                        n, graph, graph_params, env, T, g, k, args.t,
                        regret_dict_base])
                process.daemon = True
                process.start()
                threads.append(process)

        for process in threads:
            process.join()

        regret_dict = {}
        for k in [
                'ftl_degree',
                'ftl_distance',
                'ftl_bc',
                'decentralized']:
            regret_dict[k] = {}

        for elem in regret_dict_base:
            k, g, nn, gt, regrets = elem
            if nn not in regret_dict[k]:
                regret_dict[k][nn] = regrets

        dt = '{date:%Y-%m-%d_%H:%M:%S}.pkl'.format(
            date=datetime.datetime.now())
        output_file = 'exp2_%s_%s' % (args.t, dt)
        pkl.dump(
            (regret_dict, args.t), open(output_file, 'wb'),
            protocol=pkl.HIGHEST_PROTOCOL)

        # averaging now
        # colors = ['red', 'green', 'blue', 'cyan']
        # for algorithm in range(4):
        #     diam_vals = sorted(regret_dict[k].keys())
        #     plot_avg = [1.0/n*np.sum(
        #         regret_dict[algorithm][x][0]) for x in diam_vals]
        #     plot_max = [np.sum(
        #         regret_dict[algorithm][x][1]) for x in diam_vals]
        #     plot_min = [np.sum(
        #         regret_dict[algorithm][x][2]) for x in diam_vals]
        #     plt.plot(diam_vals, plot_avg, color=colors[algorithm], marker='o')
        #     plt.plot(diam_vals, plot_max, color=colors[algorithm], marker='+')
        #     plt.plot(diam_vals, plot_min, color=colors[algorithm], marker='.')

    # g = 3
    # T = 800
    # K = 20
    # print(env.means)
    #
    # graph = nx.gnp_random_graph(n, 0.05)
    # pos = nx.spring_layout(graph)
    #
    # plt.figure()
    #
    # graph_params = {}
    # graph_params['node_weight'] = 'degree'
    # graph_params['choice_type'] = 'degree'
    #
    # alg_type = 'ftl'
    # manager = Manager(n, g, alg_type, graph=graph, graph_params=graph_params)
    # print('graph partitioned')
    # manager.create_agents(env)
    # regrets1 = manager.run(T, env)
    #
    # alg_type = 'ftl'
    # graph_params['node_weight'] = 'distance'
    # manager = Manager(n, g, alg_type, graph=graph, graph_params=graph_params)
    # manager.create_agents(env)
    # regrets2 = manager.run(T, env)
    #
    # alg_type = 'ftl'
    # graph_params['node_weight'] = 'bc'
    # manager = Manager(n, g, alg_type, graph=graph, graph_params=graph_params)
    # manager.create_agents(env)
    # regrets3 = manager.run(T, env)
    #
    # alg_type = 'decentralized'
    # graph_params['node_weight'] = 'bc'
    # manager = Manager(n, g, alg_type, graph=graph, graph_params=graph_params)
    # manager.create_agents(env)
    # regrets4 = manager.run(T, env)
    #
    # plt.plot(range(T), regrets1)
    # plt.plot(range(T), regrets2)
    # plt.plot(range(T), regrets3)
    # plt.plot(range(T), regrets4)
