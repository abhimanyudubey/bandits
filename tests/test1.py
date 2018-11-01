import sys
import os
sys.path.insert(0, '..')
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# import numpy as np
import random
import numpy as np
import bandits

n_arms = 10
n_iter = 1000
n_trials = 20
regret1 = []
regret2 = []

for trial in range(n_trials):
    env = bandits.envs.Environment(n_arms)
    agent2 = bandits.agents.RandomAgent(n_arms)
    agent1 = bandits.agents.UCB1Agent(n_arms)
    sum_rewards1 = 0.0
    sum_rewards2 = 0.0

    for i in range(n_iter):

        a_i1 = agent1.play()
        a_i2 = agent2.play()

        r_i1 = env.iter(a_i1)
        r_i2 = env.iter(a_i2)

        agent1.update(r_i1)
        agent2.update(r_i2)

        sum_rewards1 += r_i1
        sum_rewards2 += r_i2

    regret1.append(env.params[env.best_arm]*n_iter-sum_rewards1)
    regret2.append(env.params[env.best_arm]*n_iter-sum_rewards2)


print('Average Regret of UCB1: %.2f' % (np.mean(regret1)))
print('Average Regret of Random: %.2f' % (np.mean(regret2)))
