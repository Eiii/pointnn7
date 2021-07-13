import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle

from pathlib import Path
from collections import defaultdict
import numpy as np
import sklearn.metrics

from . import common


def losses():
    base = Path('./cluster/final')
    losses = base.glob('loss*.pkl')
    all_data = {}
    for ls in losses:
        nn = ls.name[5:]
        nn = nn[:nn.find('.pkl')]
        drop = float(nn)
        with ls.open('rb') as fd:
            data = pickle.load(fd)
            all_data.update({(drop, k):v for k,v in data.items()})

def make_table(data):
    all_nets = list(data.keys())
    all_drops = sorted(list(data[all_nets[0]].keys()))
    # Header
    all_rows = []
    for n in all_nets:
        row = []
        row.append(n)
        for d in all_drops:
            aucs = []
            dat = data[n][d]
            for xx in data[n][d]:
                tpr, fpr = zip(*xx)
                aucs.append(sklearn.metrics.auc(fpr, tpr))
            auc = np.mean(aucs)
            err = np.std(aucs)/np.sqrt(len(aucs))
            if len(aucs) > 1:
                row.append(f'{auc:.4f} +- {err:.4f}')
            else:
                row.append(f'{auc:.4f}')
        all_rows.append(row)
    for r in all_rows:
        print(' & '.join(r) + ' \\')


def roc():
    base = Path('./cluster/final')
    files = base.glob('aucs*.pkl')
    all_data = defaultdict(dict)
    for roc_file in files:
        n = roc_file.name
        drop = float(n[n.find('-')+1:n.rfind('.')])
        with roc_file.open('rb') as fd:
            data = pickle.load(fd)
        for k in data:
            all_data[k][drop] = data[k]
    make_table(all_data)
    for net, d in all_data.items():
        plot_roc(net, d)

COLORS = ['black', 'orange', 'green', 'red']
def plot_roc(net_name, data):
    drops = sorted(list(data.keys()))
    fig = plt.figure(figsize=(8, 8), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0,1], [0,1], alpha=0.25, c='black', linestyle='--')
    for d, c in zip(drops, COLORS):
        tpr, fpr = zip(*data[d][0])
        ax.plot(fpr, tpr, c=c, label=d, linewidth=5, alpha=0.7)
        ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0.25, 0.5, 0.75])
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    figlegend = plt.figure(figsize=(3, 8), dpi=200)
    figlegend.legend(*ax.get_legend_handles_labels(), loc='center left',
                     title='Input Data\nRemoved', fontsize=30, title_fontsize=30)
    figlegend.savefig("legend.png")
    plt.tight_layout()
    fig.savefig(f'{net_name}.png')

if __name__=='__main__':
    common.config_mpl()
    roc()
