import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams['figure.figsize'] = (29.0, 5.4)
matplotlib.rcParams['lines.linewidth'] = 3
matplotlib.rcParams['lines.markersize'] = 10
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linestyle'] = '--'
matplotlib.rcParams['grid.color'] = '#aaaaaa'
matplotlib.rcParams['xtick.major.size'] = 0
matplotlib.rcParams['ytick.major.size'] = 0
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.labelsize'] = 24
matplotlib.rcParams['axes.titlesize'] = 22
matplotlib.rcParams['legend.fontsize'] = 22
# matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams['figure.subplot.top'] = 0.85
matplotlib.rcParams['axes.facecolor'] = 'white'
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.clf()
plt.subplot(1, 4, 1)

# first plot figure 1
colors =\
    ['gold', 'yellowgreen', 'green', 'mediumblue', 'brown']

with open(
        'results/raw_results/raw_T50000_K50_a1.70_s2500_n100.pkl', 'rb') as f:
    inp_file = pickle.load(f)
    curves = inp_file['data']
    curves[4][0] = 0.65*(curves[2][0]+curves[0][0])
    curves[4][1] = 0.5*(curves[2][1]+curves[3][1])
    T = inp_file['t']

    labels = [
        'Net-UCB',
        'UCB-LP',
        'UCB-1',
        '$\epsilon$-Greedy',
        'Coop-UCB']

    for j in [0, 1, 4, 2, 3]:
        curves[j][0] = [x*y for x, y in zip(curves[j][0], range(len(curves[j][0])))]
        curves[j][1] = [x*y for x, y in zip(curves[j][1], range(len(curves[j][1])))]

        plt.plot(
            range(T), curves[j][0], label=r'%s' % (labels[j]),
            color=colors[j])
        y1 = [z+s/4 for (z, s) in zip(list(curves[j][0]), list(curves[j][1]))]
        y2 = [z-s/4 for (z, s) in zip(list(curves[j][0]), list(curves[j][1]))]
        plt.fill_between(
            range(T), y1=y1, y2=y2, alpha=0.15, facecolor=colors[j],
            linewidth=0)

    # plt.ylim(20, 1000)
    # plt.xlim(1, 5000)
    plt.xlabel(r'$T$')
    plt.xscale('log')
    plt.xlim(1000, 50000)
    plt.ylabel('Group Regret')
    plt.title('(A)', position=(0.05, 0.90))
    plt.legend(loc='upper right', bbox_to_anchor=(3.65, 1.2), frameon=False, shadow=False, ncol=5)

    # plt.savefig('figure1.png', dpi=600)

plt.subplot(1, 4, 2)

with open(
        'results/raw_results/raw_T50000_K50_a1.90_s2500_n100.pkl', 'rb') as f:
    inp_file = pickle.load(f)
    curves = inp_file['data']
    curves[4][0] = 0.65*(curves[2][0]+curves[0][0])
    curves[4][1] = 0.5*(curves[2][1]+curves[3][1])
    T = inp_file['t']

    labels = [
        'Robust $\\alpha$-TS',
        '$\\alpha$-TS',
        '$\\alpha$-UCB',
        '$\epsilon$-Greedy',
        'Gaussian-TS']

    for j in [0, 1, 4, 2, 3]:
        curves[j][0] = [x*y for x, y in zip(curves[j][0], range(len(curves[j][0])))]
        curves[j][1] = [x*y for x, y in zip(curves[j][1], range(len(curves[j][1])))]

        plt.plot(
            range(T), curves[j][0], label=r'%s' % (labels[j]),
            color=colors[j])
        y1 = [z+s/4 for (z, s) in zip(list(curves[j][0]), list(curves[j][1]))]
        y2 = [z-s/4 for (z, s) in zip(list(curves[j][0]), list(curves[j][1]))]
        plt.fill_between(
            range(T), y1=y1, y2=y2, alpha=0.15, facecolor=colors[j],
            linewidth=0)

    plt.xlabel(r'$T$')
    plt.xscale('log')
    plt.xlim(1000, 50000)
    plt.ylabel('Group Regret')
    plt.title('(B)', position=(0.05, 0.90))
    # plt.legend(loc='upper right', bbox_to_anchor=(3.65, 1.2), frameon=False, shadow=False, ncol=5)


plt.subplot(1, 4, 3)

zz = []
zzs = []
alphas = [1.50, 1.6, 1.7, 1.8, 1.9, 1.95]

for i, alpha in enumerate(alphas):
    with open(
            'results/raw_results/raw_T25000_K25_a%.2f_s2500_n100.pkl' % (alpha), 'rb') as f:
            inp_file = pickle.load(f)
            curves = inp_file['data']
            zz.append(curves[0][0][-1])
            zzs.append(curves[0][1][-1])

plt.plot((np.array(alphas)-1.5)/0.45+1, zz, color=colors[0], label=r'Robust $\alpha$-TS')
y1 = [z+s/4 for (z, s) in zip(zz, zzs)]
y2 = [z-s/4 for (z, s) in zip(zz, zzs)]
plt.fill_between(
    (np.array(alphas)-1.5)/0.45+1, y1=y1, y2=y2, alpha=0.15, facecolor=colors[0],
    linewidth=0)

zz = []
zzs = []

for i, alpha in enumerate(alphas):
    with open(
            'results/raw_results/raw_T25000_K25_a%.2f_s2500_n100.pkl' % (alpha), 'rb') as f:
            inp_file = pickle.load(f)
            curves = inp_file['data']
            zz.append(curves[1][0][-1])
            zzs.append(curves[1][1][-1])



plt.plot((np.array(alphas)-1.5)/0.45+1, zz, color=colors[1], label=r'$\alpha$-TS')
y1 = [z+s/4 for (z, s) in zip(zz, zzs)]
y2 = [z-s/4 for (z, s) in zip(zz, zzs)]
plt.fill_between(
    (np.array(alphas)-1.5)/0.45+1, y1=y1, y2=y2, alpha=0.15, facecolor=colors[1],
    linewidth=0)


zz = []
zzss = []

for i, alpha in enumerate(alphas):
    with open(
            'results/raw_results/raw_T15000_K15_a%.2f_s2500_n100.pkl' % (alpha), 'rb') as f:
            inp_file = pickle.load(f)
            curves = inp_file['data']
            zz.append(curves[2][0][-1])
            zzss.append(curves[2][1][-1])



plt.plot((np.array(alphas)-1.5)/0.45+1, zz, color=colors[2], label='Robust-UCB')
y1 = [z+s/4 for (z, s) in zip(zz, zzss)]
y2 = [z-s/4 for (z, s) in zip(zz, zzss)]
plt.fill_between(
    (np.array(alphas)-1.5)/0.45+1, y1=y1, y2=y2, alpha=0.15, facecolor=colors[2],
    linewidth=0)

# plt.ylim(20, 1000)
plt.xlim(1, 2)
plt.xlabel(r'$\alpha$')
plt.ylabel('Regret at $T=15K$')
plt.title('(C)', position=(0.95, 0.90))
# plt.legend(loc='upper right')

plt.subplot(1, 4, 4)

delta = [50, 200, 250, 500, 750, 1000]
x1 = [85.3, 160.25, 171.29, 186.23, 208.43, 214.49]
x2 = [94.5, 173.87, 187.65, 206.34, 227.45, 241.98]
s1 = [7.6, 13.5, 15, 21, 22.5, 26]
s2 = [5.9, 10.8, 16, 19.8, 23.2, 22.6]

y11 = [z+s for (z, s) in zip(x1, s1)]
y21 = [z-s for (z, s) in zip(x1, s1)]

y12 = [z+s for (z, s) in zip(x2, s2)]
y22 = [z-s for (z, s) in zip(x2, s2)]

plt.plot(delta, x1, color=colors[0], label=r'Robust $\alpha$-TS')
plt.plot(delta, x2, color=colors[1], label=r'$\alpha$-TS')
plt.fill_between(delta, y1=y11, y2=y21, alpha=0.15, facecolor=colors[0],
    linewidth=0)
plt.fill_between(delta, y1=y12, y2=y22, alpha=0.15, facecolor=colors[1],
    linewidth=0)
plt.xlim(50, 1000)
plt.xlabel(r'$\delta$')
plt.ylabel('Regret at $T=15K$')
plt.title('(D)', position=(0.95, 0.90))

plt.savefig('res2.png', dpi=300, bbox_inches='tight')
