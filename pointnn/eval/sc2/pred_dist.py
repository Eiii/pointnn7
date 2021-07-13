from ...data.starcraft import StarcraftDataset, collate, collate_voxelize
from ...problem.starcraft import sc2_frame_loss, unit_count
from .. import common
from . import plot

import torch
from torch.utils.data import DataLoader

import math
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
    fixed1 = ('Fixed1', np.array([1, 2, 3]), 1)
    fixed2 = ('Fixed2', np.array([1, 4, 8]), 1)
    x = np.arange(1, 11)
    y = np.ones_like(x) / len(x)
    uni = ('Uniform', x, y)
    y = [S.norm.pdf(x, scale=4) for x in range(len(x))]
    y = [v/sum(y) for v in y]
    tail = ('Tail', x, y)
    pairs = (fixed1, fixed2, uni, tail)
    fig, axs = plt.subplots(len(pairs), 1, dpi=200, figsize=(5, 4),
                            sharex='col')
    for ax, pair in zip(axs, pairs):
        _dist_plot(ax, *pair)
    plt.tight_layout()
    fig.savefig(f'dist_example.{out_type}')

def _dist_plot(ax, title, xs, ys):
    ax.set_ylabel(title, rotation=0, labelpad=20, fontsize='large')
    ax.bar(xs, ys, color='black')
    ax.set_xlim(0.5, 15)
    if not isinstance(ys, int):
        ax.set_ylim(0, max(ys)*1.2)

def pred_loss(net, ds, batch_size, collate_fn):
    with torch.no_grad():
        net = common.make_net(net)
        net = net.eval().cuda()
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
        total_loss = 0
        total_unit = 0
        all_losses = []
        all_times = []
        for item in itertools.islice(loader, None):
            item = seq_to_cuda(item)
            args = net.get_args(item)
            pred = net(*args)
            loss_matrix = sc2_frame_loss(item, pred)
            mask = item['pred_ts_mask'].unsqueeze(-1) * item['pred_ids_mask'].unsqueeze(-2)
            ts = torch.arange(1, 15+1).view(1, -1, 1)
            loss_entries = torch.masked_select(loss_matrix, mask)
            times = torch.masked_select(ts, mask)
            all_losses.append(loss_entries)
            all_times.append(times)
        return torch.cat(all_losses), torch.cat(all_times)

def main(nets, data_path, batch_size, out_type):
    ts = list(range(1, 15+1))
    default_args = {'base': data_path, 'max_files': 1, 'max_hist': 5,
                    'num_pred': len(ts), 'hist_dist': 'fixed', 'hist_dist_args': {'ts': [0, -1, -2, -3, -4]},
                    'frame_skip': 10}
    nets = common.load_any(nets)
    arg = {'pred_dist': 'fixed', 'pred_dist_args': {'ts': ts}}
    loss_dict = defaultdict(list)
    for net in nets:
        if 'Mink' in net['net_type']:
            r = 0.05
            vres = [r, r, 1]
            collate_fn = functools.partial(collate_voxelize, vres)
        else:
            collate_fn = collate
        ds = StarcraftDataset(**default_args, **arg)
        loss_pair = pred_loss(net, ds, batch_size, collate_fn)
        loss_dict[net['measure'].name].append(loss_pair)
    with open('tpc_pred_dist_results.pkl', 'wb') as fd:
        pickle.dump(loss_dict, fd)
    plot_dist_array(loss_dict, out_type)

def main_fast(out_type):
    with open('tpc_pred_dist_results.pkl', 'rb') as fd:
        loss_dict = pickle.load(fd)
    plot_dist_array(loss_dict, out_type)

def invert_dict(d):
    out = defaultdict(dict)
    for x in d:
        for y in d[x]:
            out[y][x] = d[x][y]
    return out

def plot_dist_array_im(loss_dict):
    fig, ax = plt.subplots(dpi=200, figsize=(10, 10))
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
    types = ['TPC']
    dists = ['Fixed1', 'Fixed2', 'Uniform', 'Tail']
    for i, net_type in enumerate(types):
        fig, axs = plt.subplots(len(dists), 1, dpi=200, figsize=(5, 4),
                                sharex='col', sharey='col',
                                subplot_kw={'yscale': 'log'})
        for j, dist in enumerate(dists):
            name = f'{net_type}-{dist}'
            pairs = loss_dict[name]
            losses = torch.cat([p[0] for p in pairs])
            times = torch.cat([p[1] for p in pairs])
            xs = []
            ys = []
            errs = []
            for t in times.unique():
                t_mask = (times == t)
                t_losses = torch.masked_select(losses, t_mask)
                mean = t_losses.mean()
                stdmean = t_losses.std() / math.sqrt(t_losses.numel())
                xs.append(t.item())
                ys.append(mean.item())
                errs.append(stdmean.item())
            ax = axs[j]
            ax.bar(xs, ys, align='center', color='black')
            if j == 0:
                ax.set_title('Temporal PointConv')
            if i == 0:
                ax.set_ylabel(dist, rotation=0, labelpad=20)
    plt.tight_layout()
    fig.savefig(f'array.{out_type}')


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nets', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--output', default='png')
    return parser

def seq_to_cuda(d):
    if isinstance(d, dict):
        return {k:v.cuda() for k,v in d.items()}
    elif isinstance(d, list):
        return [v.cuda() for v in d]

if __name__ == '__main__':
    args = make_parser().parse_args()
    dist_examples(args.output)
    if args.fast:
        main_fast(args.output)
    else:
        main(args.nets, args.data, args.bs, args.output)
