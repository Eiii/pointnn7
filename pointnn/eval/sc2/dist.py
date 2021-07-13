from ...data.starcraft import StarcraftDataset, collate, collate_voxelize
from ...problem.starcraft import sc2_frame_loss, unit_count
from .. import common
from . import plot

import torch
from torch.utils.data import DataLoader

import itertools
import functools
import argparse
import pickle
import numpy as np
import scipy.stats as S
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def dist_examples(out_type):
    fig = plt.figure(figsize=(5, 4), dpi=200)
    fixed1 = ('Fixed1', np.array([0, -1, -2, -3, -4]), 1)
    fixed2 = ('Fixed2', np.array([0, -2, -4, -7, -10]), 1)
    x = np.arange(0, -11, -1)
    y = np.ones_like(x) / len(x)
    uni = ('Uniform', x, y)
    y = [S.norm.pdf(x, scale=4) for x in range(len(x))]
    y = [a/sum(y) for a in y]
    tail = ('Tail', x, y)
    pairs = (fixed1, fixed2, uni, tail)
    for idx, pair in enumerate(pairs):
        ax = fig.add_subplot(len(pairs), 1, idx+1)
        _dist_plot(ax, *pair)
    plt.tight_layout()
    fig.savefig(f'dist_example.{out_type}')

def _dist_plot(ax, title, xs, ys):
    ax.set_ylabel(title, rotation=0, labelpad=15, fontsize='large')
    ax.bar(xs, ys, color='black')
    ax.set_xlim(-11, 1)
    if not isinstance(ys, int):
        ax.set_ylim(0, max(ys)*1.2)

def pred_loss(net, ds, batch_size, collate_fn):
    with torch.no_grad():
        net = common.make_net(net)
        net = net.eval().cuda()
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
        total_loss = 0
        total_unit = 0
        for item in itertools.islice(loader, None):
            item = seq_to_cuda(item)
            args = net.get_args(item)
            pred = net(*args)
            total_loss += sc2_frame_loss(item, pred).sum()
            total_unit += unit_count(item)
        return (total_loss/total_unit).item()

def main(nets, data_path, batch_size):
    default_args = {'base': data_path, 'max_files': 1, 'max_hist': 5,
                    'num_pred': 3, 'pred_dist': 'fixed', 'pred_dist_args': {'ts': [1, 3, 8]}}
    nets = common.load_any(nets)
    fixed1 = ('Fixed1', {'hist_dist': 'fixed', 'hist_dist_args': {'ts': [0, -1, -2, -3, -4]}})
    fixed2 = ('Fixed2', {'hist_dist': 'fixed', 'hist_dist_args': {'ts': [0, -2, -4, -7, -10]}})
    uni = ('Uniform', {'hist_dist': 'uniform', 'hist_dist_args': {'max': 10}})
    tail = ('Tail', {'hist_dist': 'tail', 'hist_dist_args': {'max': 10}})
    dist_args = (fixed1, fixed2, uni, tail)
    loss_dict = dict()
    for dist_name, arg in dist_args:
        print(dist_name)
        dist_loss_dict = defaultdict(list)
        for net in nets:
            if 'Mink' in net['net_type']:
                r = 0.05
                vres = [r, r, 1]
                collate_fn = functools.partial(collate_voxelize, vres)
            else:
                collate_fn = collate
            ds = StarcraftDataset(**default_args, **arg)
            loss = pred_loss(net, ds, batch_size, collate_fn)
            dist_loss_dict[net['measure'].name].append(loss)
        loss_dict[dist_name] = dict(dist_loss_dict)
    with open('dist_hist_results.pkl', 'wb') as fd:
        pickle.dump(loss_dict, fd)
    plot_dist_array(loss_dict)

def main_fast(out_type):
    with open('dist_hist_results.pkl', 'rb') as fd:
        loss_dict = pickle.load(fd)
    plot_dist_array(loss_dict, out_type)

def invert_dict(d):
    out = defaultdict(dict)
    for x in d:
        for y in d[x]:
            out[y][x] = d[x][y]
    return out

def plot_dist_array_im(loss_dict):
    fig, ax = plt.subplots(dpi=200, figsize=(5, 4))
    dists = ['Fixed1', 'Fixed2', 'Uniform', 'Tail']
    types = ['TPC', 'SeFT']
    nets = [f'{a}-{b}' for a, b in itertools.product(types, dists)]
    loss_array = np.zeros((len(nets), len(dists)))
    for i, net in enumerate(nets):
        for j, dist in enumerate(dists):
            loss_array[i, j] = np.mean(loss_dict[dist][net])
    ax.imshow(np.log(loss_array), cmap='plasma')
    for i, net in enumerate(nets):
        for j, dist in enumerate(dists):
            text = '{:.2e}'.format(loss_array[i, j])
            color = 'k' if loss_array[i, j] > np.mean(loss_array) else 'w'
            ax.text(j, i, text, ha='center', va='center', color=color)
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels(dists)
    ax.set_yticks(range(len(nets)))
    ax.set_yticklabels(nets, rotation=0)
    plt.tight_layout()
    fig.savefig('array.png')

def plot_dist_array(loss_dict, out_type='png'):
    loss_dict = invert_dict(loss_dict)
    dists = ['Fixed1', 'Fixed2', 'Uniform', 'Tail']
    types = ['TPC', 'SeFT', 'Mink']
    nets = [f'{a}-{b}' for a, b in itertools.product(types, dists)]

    fig, axs = plt.subplots(len(dists), len(types), dpi=200, figsize=(5, 4),
                            sharex='col', sharey='row', subplot_kw={'yscale': 'linear'})
    for i, net_type in enumerate(types):
        for j, dist in enumerate(dists):
            name = f'{net_type}-{dist}'
            ref_name = f'TPC-{dist}'
            net_dict = loss_dict[name]
            mean_fn = np.mean
            ref = mean_fn(loss_dict[ref_name][dist])
            norm_vals = {k:np.array(v)/ref for k,v in net_dict.items()}
            norm_means = {k:mean_fn(v) for k,v in norm_vals.items()}
            norm_stds = {k:np.std(v) for k,v in norm_vals.items()}
            means = [norm_means[dist] for dist in dists]
            stds = [norm_stds[dist] for dist in dists]
            xs = np.arange(len(means))
            ax = axs[j, i]
            cols = ['red' if dist == other_dist else 'black' for other_dist in dists]
            ax.bar(xs, means, align='center', color=cols)
            ax.set_xticks(xs)
            ax.set_xticklabels(dists, rotation=45)
            ax.set_yticklabels([])
            if j == 0:
                ax.set_title(net_type)
            if i == 0:
                ax.set_ylabel(dist, rotation=0, labelpad=20)
    plt.tight_layout()
    fig.savefig(f'array.{out_type}')


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nets', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--format', default='png')
    return parser

def seq_to_cuda(d):
    if isinstance(d, dict):
        return {k:v.cuda() for k,v in d.items()}
    elif isinstance(d, list):
        return [v.cuda() for v in d]

if __name__ == '__main__':
    args = make_parser().parse_args()
    dist_examples(args.format)
    #main(args.nets, args.data, args.bs)
    main_fast(args.format)
