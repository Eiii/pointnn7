from ...data.starcraft import StarcraftDataset, collate
from ...problem.starcraft import sc2_frame_loss, get_pred_ts, get_alive

from .. import common
from ... import utils

import math
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from collections import defaultdict

import pickle

import matplotlib.pyplot as plt


def seq_to_device(d, device):
    if isinstance(d, dict):
        return {k: v.to(device) for k, v in d.items()}
    elif isinstance(d, list):
        return [v.to(device) for v in d]


def pred_safe(model, ds, bs, device='cpu'):
    while bs != 1:
        print(f'batch size={bs}')
        try:
            return pred(model, ds, bs, device)
        except RuntimeError as e:
            print(e)
            bs = bs//2
    raise RuntimeError()


def pred(model, ds, bs, device='cpu'):
    losses = []
    pred_ts = []
    alive = []
    loader = DataLoader(ds, batch_size=bs, collate_fn=collate, num_workers=8)
    for item in loader:
        item = seq_to_device(item, device)
        args = model.get_args(item)
        pred = model(*args)
        ls = sc2_frame_loss(item, pred, reduction='none')
        ls = ls.detach()
        flat_pred_ts = get_pred_ts(item)
        flat_alive = get_alive(item)
        losses.append(ls)
        alive.append(flat_alive)
        pred_ts.append(flat_pred_ts)
    result = {'losses': torch.cat(losses).cpu(),
              'ts': torch.cat(pred_ts).cpu(),
              'alive': torch.cat(alive).bool().cpu()}
    return result


def run_net(net, ds, max_bs):
    with torch.no_grad():
        net = net.eval().cuda()
        return pred_safe(net, ds, max_bs, device='cuda')


def make_dataset(data_path, ts):
    if ts is None:
        ts = [1, 2, 4, 7]
    ds = StarcraftDataset(data_path, num_pred=len(ts), max_hist=5, hist_dist='uniform',
                          hist_dist_args={'max': 10}, pred_dist='fixed',
                          pred_dist_args={'ts': ts}, frame_skip=1)
    return ds


def batch(net_paths, ds, max_bs, out='sc-loss.pkl'):
    print(len(ds))
    loss_dict = {}
    for net_path in net_paths:
        print(net_path)
        net = common.make_net(common.load_result(net_path))
        pred_result = run_net(net, ds, max_bs)
        loss_dict[net_path] = pred_result
        with open(out, 'wb') as fd:
            pickle.dump(loss_dict, fd)


def arch_type(path):
    with Path(path).open('rb') as fd:
        return pickle.load(fd)['net_type']


def net_type(path):
    return Path(path).stem.split(':')[0]


def param_count(path):
    nt = net_type(path)
    lut = {'Small': 100, 'Med': 1000, 'Large': 5000}
    for k, v in lut.items():
        if k in nt:
            return v
    return -1


def group_result(paths):
    result = {}
    data = {}
    for p in paths:
        with Path(p).open('rb') as fd:
            d = pickle.load(fd)
            data.update(d)
    net_types = {net_type(p) for p in data}
    for n_type in net_types:
        net_data = {k:v for k,v in data.items() if net_type(k)==n_type}
        net_counts = {k:param_count(k) for k in net_data}
        for net_count in set(net_counts.values()):
            ns = {n for n,c in net_counts.items() if c==net_count}
            lcl_nets = {k:v for k,v in data.items() if k in ns}
            losses = [x['losses'] for x in lcl_nets.values()]
            ts = [x['ts'] for x in lcl_nets.values()]
            alive = [x['alive'] for x in lcl_nets.values()]
            all_loss = torch.cat(losses, dim=0)
            all_t = torch.cat(ts, dim=0)
            all_alive = torch.cat(alive, dim=0)
            result[(n_type, net_count)] = {'loss': all_loss, 't': all_t, 'alive': all_alive}
    return result

def plot(paths):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    result = group_result(paths)
    net_types = {k[0] for k in result}
    for net_type in net_types:
        counts = {k[1] for k in result if k[0]==net_type}
        plot_data = []
        for count in sorted(counts):
            d_key = (net_type, count)
            loss = result[d_key]['loss'].mean(dim=-1)
            ts = result[d_key]['t']
            alive = result[d_key]['alive']
            mean = loss.mean()
            mean = min(mean, 0.5)
            std = loss.std() / math.sqrt(len(loss))
            plot_data.append((count, mean, std))
            #
            logloss = loss.log()
            uniq_ts = ts.unique().tolist()
            hist_xs = []
            for t in uniq_ts:
                mask = (ts == t)
                hist_xs.append(logloss[mask].tolist())
            fig2, ax2 = plt.subplots(figsize=(8,5), dpi=200)
            ax2.hist(hist_xs, bins=250,
                     histtype='stepfilled', stacked=True,
                     label=[str(int(t)) for t in uniq_ts])
            ax2.legend()
            fig2.savefig(f'{net_type}-{count}-t.png')
            #
            uniq_alive = alive.unique().tolist()
            hist_as = []
            for a in uniq_alive:
                mask = (alive == a)
                hist_as.append(logloss[mask].tolist())
            fig3, ax3 = plt.subplots(figsize=(8,5), dpi=200)
            ax3.hist(hist_as, bins=250,
                     histtype='stepfilled', stacked=True,
                     label=[str(int(t)) for t in uniq_alive])
            ax3.legend()
            fig3.savefig(f'{net_type}-{count}-alive.png')
        xs, ys, errs = zip(*plot_data)
        ax.errorbar(xs, ys, label=net_type, fmt='x:')
    ax.set_yscale('log')
    fig.legend()
    fig.savefig('out.png')


def plot_old(path):
    pkls = list(Path('.').glob(path+'out*.pkl'))
    print(pkls)
    ts = [int(p.stem.split('-')[-1]) for p in pkls]
    plot_data = defaultdict(dict)
    for t, pkl in sorted(zip(ts, pkls)):
        with pkl.open('rb') as fd:
            data = pickle.load(fd)
        for net, losses in data.items():
            losses = losses.sum(dim=-1)
            mean = losses.mean().item()
            mean = min(mean, 5)
            sem = losses.std().item() / (len(losses)**0.5)
            plot_data[net][t] = (mean, sem)
    fig, ax = plt.subplots()
    width = 0.9/3
    for i, (name, t_data) in enumerate(plot_data.items()):
        ts = list(range(1, 1+12))
        means = [t_data[t][0] for t in ts]
        print(means)
        plot_ts = [t+i*width for t in ts]
        ax.bar(plot_ts, means, width, label=name)
    ax.legend()
    #ax.set_yscale('log')
    ax.set_xlabel('Time Delta')
    ax.set_ylabel('Avg. Loss')
    fig.savefig('out.png')


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/sc2scene')
    parser.add_argument('--t', default=None, type=int)
    parser.add_argument('--out', default='sc-loss.pkl')
    parser.add_argument('--net', nargs='+')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--bs', type=int, default=32)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    if args.plot:
        plot(args.net)
    else:
        ts = [args.t] if args.t is not None else None
        ds = make_dataset(args.data, ts)
        batch(args.net, ds, args.bs, out=args.out)
