import numpy as np
from scipy.special import gamma
from scipy.stats import norm
import copy
import matplotlib
from pathos.multiprocessing import Pool
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle as pickle

matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linestyle'] = '--'
matplotlib.rcParams['grid.color'] = '#aaaaaa'
matplotlib.rcParams['xtick.major.size'] = 0
matplotlib.rcParams['ytick.major.size'] = 0
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['legend.fontsize'] = 14
# matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams['figure.subplot.top'] = 0.85
matplotlib.rcParams['axes.facecolor'] = 'white'
matplotlib.rcParams['axes.linewidth'] = 0.8
Q = 50


def cms_alpha(_alpha, _beta, mu, sigma):
    ''' Generate a random sample from the alpha-stable distribution f
        using the CMS Algorithm, where f : S_alpha(beta, mu, sigma) '''

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


def alpha_ts_posterior_update(
        prior_mu, prior_sigma, prior_lambda_sum, reward,
        alpha, sigma, prior_mu_sample, robust=False):
    ''' Update the posterior distribution of the alpha-stable rewards given the
        earlier mean and variance using the conditional SMiN representation'''

    for q in range(Q):
        lambda_t = cms_alpha(alpha*0.5, 1, 0, 1)
        # if not robust:
        vt = reward - prior_mu_sample
        u_fac = np.exp(-0.5)*1/(np.sqrt(2*np.pi*(vt**2)))
        u = np.random.uniform(0, u_fac)
        lim = norm.pdf(vt, loc=0, scale=np.sqrt(lambda_t)*sigma)
        i, min_u, corr_lambda = 0, u, lambda_t

        while u > lim:
            lambda_t = cms_alpha(alpha*0.5, 1, 0, 1)
            u = np.random.uniform(0, u_fac)
            lim = norm.pdf(vt, loc=0, scale=np.sqrt(lambda_t)*sigma)
            if u < min_u or u <= lim:
                corr_lambda = lambda_t
            i += 1
            if i > 100000:
                break

        lambda_t = corr_lambda
        this_mu = (
            prior_mu * prior_lambda_sum +
            reward/lambda_t)/(prior_lambda_sum + 1/lambda_t)
        this_sigma = prior_sigma * \
            prior_lambda_sum / (prior_lambda_sum + 1/lambda_t)
        lambda_sum = prior_lambda_sum + 1/lambda_t
        prior_mu_sample = np.random.normal(this_mu, np.sqrt(this_sigma))

    return this_mu, this_sigma, lambda_sum


def alpha_ts(K, T, alpha, sigma, mus, mu_priors, robust=False):

    assert len(mus) == len(mu_priors)
    assert len(mus) == K

    sigmas = [sigma for i in range(K)]
    lambda_sums = [1 for i in range(K)]
    reward_sum = 0
    regrets = []

    for t in range(1, T+1):

        posterior_samples = [
            np.random.normal(mu_priors[i], np.sqrt(sigmas[i]))
            for i in range(K)]
        ca = np.argmax(posterior_samples)
        reward = cms_alpha(alpha, 0, mus[ca], sigma)

        reward_copy = copy.copy(reward)
        if robust:
            # calculate robust mean bound
            epsilon = alpha - 10**(-20)
            robust_bound = 2**((2+epsilon)/(1+epsilon)) * \
                gamma(1 + epsilon/2)*gamma(1 - (1+epsilon)/alpha) * \
                sigma/(gamma(0.5)*gamma((1-epsilon)/2))*t**(1/(1+epsilon)) * \
                (2*np.log(T))**(1/(1+epsilon))
            if np.abs(reward) > robust_bound:
                reward = 0

        mu_priors[ca], sigmas[ca], lambda_sums[ca] =\
            alpha_ts_posterior_update(
                mu_priors[ca], sigmas[ca],
                lambda_sums[ca], reward, alpha, sigma,
                posterior_samples[ca], robust=robust)

        reward_sum += reward_copy
        mean_regret = np.max(mus) - reward_sum/t

        regrets.append(mean_regret)

    return regrets


def robust_ucb(K, T, alpha, sigma, mus):
    ''' The Robust-UCB algorithm for heavy-tailed distributions
        using a truncated mean estimator as described in
        https://arxiv.org/pdf/1209.1727.pdf'''

    assert len(mus) == K
    epsilon = alpha - 10**(-20)
    n_t = [0]*K
    emp_mean = [0.0]*K
    reward_sum = 0
    regrets = []

    for t in range(1, T+1):
        ucb = [0]*K
        for k in range(K):
            if n_t[k] == 0:
                ucb[k] = np.inf
            else:
                ucb[k] = 2**((2+epsilon)/(1+epsilon)) * \
                    gamma(1 + epsilon/2)*gamma(1 - (1+epsilon)/alpha) * \
                    sigma/(gamma(0.5)*gamma((1-epsilon)/2))*t**(1/(1+epsilon))\
                    * (2 * np.log(T))**(1/(1+epsilon))
                ucb[k] += emp_mean[k]

        ca = np.argmax(ucb)
        reward = cms_alpha(alpha, 0, mus[ca], sigma)

        emp_mean[ca] = (emp_mean[ca]*n_t[ca] + reward)/(n_t[ca]+1)
        n_t[ca] += 1

        reward_sum += reward
        mean_regret = np.max(mus) - reward_sum/t

        regrets.append(mean_regret)

    return regrets


def epsilon_greedy(K, T, alpha, sigma, mus, epsilon=None):
    ''' Epsilon-greedy with a linearly decaying epsilon. '''

    assert len(mus) == K
    if epsilon is None:
        epsilon = 1.0/K

    n_t = [0]*K
    emp_mean = [0.0]*K
    reward_sum = 0
    regrets = []

    for t in range(1, T+1):

        epsilon_t = epsilon*(1 - t/(T-K))

        if t < K:
            ca = t
        else:
            p = np.random.uniform(0, 1)
            if p > epsilon_t:
                ca = np.argmax(emp_mean)
            else:
                ca = np.random.choice(range(K))

        reward = cms_alpha(alpha, 0, mus[ca], sigma)
        emp_mean[ca] = (emp_mean[ca]*n_t[ca] + reward)/(n_t[ca]+1)
        n_t[ca] += 1

        reward_sum += reward
        mean_regret = np.max(mus) - reward_sum/t

        regrets.append(mean_regret)

    return regrets


def random(K, T, alpha, sigma, mus):
    ''' Random agent.'''

    assert len(mus) == K

    reward_sum = 0
    regrets = []

    for t in range(1, T+1):
        ca = np.random.choice(range(K))

        reward = cms_alpha(alpha, 0, mus[ca], sigma)
        reward_sum += reward
        mean_regret = np.max(mus) - reward_sum/t

        regrets.append(mean_regret)

    return regrets


# experimental pipelines
def compare_algorithms(num_trials, K, T, alpha, sigma):

    regrets_overall = []
    trials_overall = [None]*5

    for n in range(num_trials):
        mus = list(np.random.uniform(0, 2000, (K, 1)))
        mu_priors = list(np.random.uniform(0, 2000, (K, 1)))

        this_trial = []
        this_trial.append(np.expand_dims(
            np.array(alpha_ts(K, T, alpha, 25, mus, mu_priors)), 0))
        this_trial.append(np.expand_dims(
            np.array(alpha_ts(K, T, alpha, 25, mus, mu_priors, True)), 0))
        this_trial.append(np.expand_dims(
            np.array(robust_ucb(K, T, alpha, sigma, mus)), 0))
        this_trial.append(np.expand_dims(
            np.array(epsilon_greedy(K, T, alpha, sigma, mus)), 0))
        this_trial.append(np.expand_dims(
            np.array(random(K, T, alpha, sigma, mus)), 0))

        for j in range(5):
            if trials_overall[j] is None:
                trials_overall[j] = this_trial[j]
            else:
                trials_overall[j] = np.concatenate(
                    (trials_overall[j], this_trial[j]), 0)

        if n % 5 == 0:
            print('%d trials completed.' % (n+1))

    for j in range(5):
        regrets_overall.append([
            np.mean(trials_overall[j], axis=0),
            np.std(trials_overall[j], axis=0)])

    return regrets_overall


def compare_algorithms_pool(num_trials, K, T, alpha, sigma, n_threads=32):

    regrets_overall = []
    trials_overall = [None]*5

    mpool = Pool(n_threads)
    pool_args = []

    for n in range(num_trials):
        mus = np.random.uniform(0, 2000, (K, 1)).flatten()
        d = np.random.uniform(-400, 400, (K, 1)).flatten()
        mu_priors = mus + d
        pool_args.append((
            K, T, alpha, sigma, mus.tolist(), mu_priors.tolist()))

    def map_fn(args):

        K, T, alpha, sigma, mus, mu_priors = args

        this_trial = []
        this_trial.append(np.expand_dims(
            np.array(alpha_ts(K, T, alpha, sigma, mus, mu_priors)), 0))
        this_trial.append(np.expand_dims(
            np.array(alpha_ts(K, T, alpha, sigma, mus, mu_priors, True)), 0))
        this_trial.append(np.expand_dims(
            np.array(robust_ucb(K, T, alpha, sigma, mus)), 0))
        this_trial.append(np.expand_dims(
            np.array(epsilon_greedy(K, T, alpha, sigma, mus)), 0))
        this_trial.append(np.expand_dims(
            np.array(random(K, T, alpha, sigma, mus)), 0))

        return this_trial

    results = mpool.map(map_fn, pool_args)

    for result in results:
        for j in range(5):
            if trials_overall[j] is None:
                trials_overall[j] = result[j]
            else:
                trials_overall[j] = np.concatenate(
                    (trials_overall[j], result[j]), 0)

    for j in range(5):
        regrets_overall.append([
            np.mean(trials_overall[j], axis=0),
            np.std(trials_overall[j], axis=0)])

    return regrets_overall


def plot_curves(labels, curves, T, output_file):

    colors =\
        ['green', 'blue', 'mediumvioletred', 'darkorchid', 'purple']

    for j in range(len(labels)):
        plt.plot(
            range(T), curves[j][0], label=r'%s' % (labels[j]),
            color=colors[j])
        y1 = [z+s/2 for (z, s) in zip(list(curves[j][0]), list(curves[j][1]))]
        y2 = [z-s/2 for (z, s) in zip(list(curves[j][0]), list(curves[j][1]))]
        plt.fill_between(
            range(T), y1=y1, y2=y2, alpha=0.15, facecolor=colors[j],
            linewidth=0)

    plt.xscale('log')
    plt.ylim(0, 2000)
    plt.xlim(100, T)
    plt.xlabel('T')
    plt.ylabel('Average Regret at Time T')
    plt.legend(loc='upper right')

    plt.savefig(output_file, dpi=300)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('alpha-Stable Bandit Simulations')
    parser.add_argument('-k', help='Number of arms', required=True, type=int)
    parser.add_argument(
        '-t', help='Number of iterations', type=int, default=None)
    parser.add_argument('-a', '--alpha', help='alpha', default=1.8, type=float)
    parser.add_argument('-s', '--sigma', help='sigma', default=2500, type=int)
    parser.add_argument(
        '-n', '--trials', help='Number of trials', default=100, type=int)
    parser.add_argument(
        '-c', '--cores', help='Number of threads', default=16, type=int)

    args = parser.parse_args()

    if args.t is None:
        args.t = args.k*1000

    labels = [
        'Alpha-TS',
        'Robust Alpha-TS',
        'Robust UCB',
        '$\epsilon$-Greedy',
        'Random']

    if args.cores == 1:
        z = compare_algorithms(
            args.trials, args.k, args.t, args.alpha, args.sigma)
    else:
        z = compare_algorithms_pool(
            args.trials, args.k, args.t, args.alpha, args.sigma, args.cores)

    data_dump = {
        'k': args.k,
        't': args.t,
        'alpha': args.alpha,
        'sigma': args.sigma,
        'data': z}
    with open('raw_T%d_K%d_a%.2f_s%d_n%d.pkl', 'wb') as out_file:
        pickle.dump(data_dump, out_file, protocol=pickle.HIGHEST_PROTOCOL)

    plot_curves(
        labels, z, args.t, 'comparsion_T%d_K%d_a%.2f_s%d_n%d.pdf' %
        (args.t, args.k, args.alpha, args.sigma, args.trials))
