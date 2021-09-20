from ...data.starcraft import StarcraftDataset, collate
from ...problem.starcraft import sc2_frame_loss

from .. import common

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


def pred(model, ds, device='cpu'):
    losses = []
    loader = DataLoader(ds, batch_size=64, collate_fn=collate, num_workers=8)
    for item in loader:
        item = seq_to_device(item, device)
        args = model.get_args(item)
        pred = model(*args)
        ls = sc2_frame_loss(item, pred, reduction='none')
        ls = ls.detach()
        losses.append(ls)
    return torch.cat(losses)


def run_net(net, ds):
    with torch.no_grad():
        net = net.eval().cuda()
        return pred(net, ds, device='cuda')


def show_results(losses):
    mean_preds = losses.mean(dim=0)
    print(mean_preds.tolist())
    mean = mean_preds.mean()
    print(mean.tolist())
    return mean


def batch(data_path, path, ts=[1, 2, 4, 7], out='sc-loss.pkl'):
    names = ['TInt-Small', 'TPC-Small']
    ds = StarcraftDataset(data_path, num_pred=len(ts), max_hist=5, hist_dist='uniform',
                          hist_dist_args={'max': 10}, pred_dist='fixed',
                          pred_dist_args={'ts': ts},
                          frame_skip=5)
    print(len(ds))
    loss_dict = {}
    for name in names:
        print(name)
        net_paths = path.glob(name+'*.pkl')
        res_l = []
        for net_path in net_paths:
            print('x')
            net = common.make_net(common.load_result(net_path))
            losses = run_net(net, ds)
            res_l.append(losses)
        all_losses = torch.cat(res_l)
        loss_dict[name] = all_losses.cpu()
    with open(out, 'wb') as fd:
        pickle.dump(loss_dict, fd)


def table(path):
    with open(path, 'rb') as fd:
        data = pickle.load(fd)
    for key in data:
        all_losses = data[key]
        means = all_losses.mean(dim=0)
        stds = all_losses.std(dim=0)
        count = all_losses.size(0)
        strs = []
        for m, s in zip(means, stds):
            m = m.item()
            s = s.item()
            strs.append(f'$ {m:.3f} \\var{{{s:.4f}}} $')
        total = all_losses.sum(-1)
        tot_mean = total.mean()
        tot_err = 1.96 * total.std() / math.sqrt(count)
        strs.append(f'$ {tot_mean:.3f} \\var{{{tot_err:.4f}}} $')
        print(' & '.join(strs) + ' \\\\')


def plot(path):
    pkls = list(Path('.').glob(path+'*'))
    ts = [int(p.stem.split('-')[-1]) for p in pkls]
    plot_data = defaultdict(dict)
    for t, pkl in sorted(zip(ts, pkls)):
        with pkl.open('rb') as fd:
            data = pickle.load(fd)
        for net, losses in data.items():
            losses = losses.sum(dim=-1)
            mean = losses.mean().item()
            sem = losses.std().item() / (len(losses)**0.5)
            plot_data[net][t] = (mean, sem)
    fig, ax = plt.subplots()
    width = 0.4
    for i, (name, t_data) in enumerate(plot_data.items()):
        ts = list(range(1, 1+10))
        means = [t_data[t][0] for t in ts]
        plot_ts = [t+i*width for t in ts]
        ax.bar(plot_ts, means, width, label=name)
    ax.legend()
    ax.set_xlabel('Time Delta')
    ax.set_ylabel('Avg. Loss')
    fig.savefig('out.png')


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/sc2scene')
    parser.add_argument('--t', default=None, type=int)
    parser.add_argument('--out', default='sc-loss.pkl')
    parser.add_argument('--net', default=Path('./output/sc2demo'), type=Path)
    parser.add_argument('--table', action='store_true')
    parser.add_argument('--plot', action='store_true')
    return parser


if __name__=='__main__':
    args = make_parser().parse_args()
    if args.table:
        table(args.data)
    elif args.plot:
        plot(args.data)
    else:
        ts = [args.t] if args.t is not None else None
        batch(args.data, args.net, ts=ts, out=args.out)
