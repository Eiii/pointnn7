from ...data.weather import WeatherDataset, Anomaly, collate, collate_voxelize
from ...problem.weather import scaled_loss
from .baselines import KNN, SelfEnsemble

from ..common import load_result, make_net

import argparse
import functools
import pickle
import itertools
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics

def seq_to_device(d, device):
    if isinstance(d, dict):
        return {k:v.to(device) for k,v in d.items()}
    elif isinstance(d, list):
        return [v.to(device) for v in d]

def pred_losses(model, ds, voxel_res=None, device='cpu'):
    losses = []
    anoms = []
    if voxel_res:
        collate_fn = functools.partial(collate_voxelize, voxel_res)
    else:
        collate_fn = collate
    loader = DataLoader(ds, batch_size=64, collate_fn=collate_fn, num_workers=8)
    for item in loader:
        item = seq_to_device(item, device)
        args = model.get_args(item)
        pred = model(*args)
        ls = scaled_loss(ds.ds, item['target'], pred).detach()
        ls = ls.sum(dim=-1).view(-1)
        losses.append(ls)
        anoms.append(item['is_anom'].view(-1))
    return torch.cat(losses), torch.cat(anoms)

def thresh_apply(losses, anoms, t):
    flagged = losses > t
    notflagged = losses < t
    real = anoms
    notreal = ~anoms
    tp = (flagged&real).sum().float()
    fp = (flagged&notreal).sum().float()
    tn = (notflagged&notreal).sum().float()
    fn = (notflagged&real).sum().float()
    tpr = tp / (tp+fn)
    fpr = fp / (tn+fp)
    return tpr, fpr

def calc_roc(losses, anoms):
    pts = []
    print(len(losses))
    for p in torch.arange(0, 1, 1/1000):
        thresh = np.quantile(losses, p)
        pts.append(thresh_apply(losses, anoms, thresh))
    print(len(pts))
    return pts

def plot_rocs(data, out_type):
    # Plot
    for net_name, pts in data.items():
        print(net_name)
        fig = plt.figure(figsize=(3, 3), dpi=200)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([0,1],[0,1], alpha=0.25, c='black')
        pts = pts[0] #TODO: Hack
        tpr, fpr = zip(*pts)
        ax.plot(fpr, tpr, label=net_name, c='black')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        auc = sklearn.metrics.auc(fpr, tpr)
        ax.set_title(f'AUC={auc:.4f}')
        fig.savefig(f'auc-{net_name}.{out_type}')

def run_knn(k, ds):
    return pred_losses(KNN(k), ds)

def run_net(net, ds, voxel_res=None):
    with torch.no_grad():
        net = net.eval().cuda()
        ens = SelfEnsemble(net, 3, {'force_dropout': 0.2})
        return [x.cpu() for x in pred_losses(ens, ds, device='cuda', voxel_res=voxel_res)]

def plot(path, out_type):
    with open(path, 'rb') as fd:
        data = pickle.load(fd)
    plot_rocs(data, out_type)

def batch(data_path, path, drop):
    names = ['TPC', 'SeFT', 'Mink']
    ds = WeatherDataset(data_path, load_train=False, drop=drop)
    anom = Anomaly(ds)
    aucs = {}
    for name in names:
        net_paths = list(path.glob(name+'*.pkl'))
        print(name, net_paths)
        all_aucs = []
        for net_path in net_paths:
            result = load_result(net_path)
            net = make_net(result)
            print(f"{result['measure'].name}:")
            kwargs = {'voxel_res': 6000} if name == 'Mink' else dict()
            out = run_net(net, anom, **kwargs)
            all_aucs.append(calc_roc(*out))
        aucs[name] = all_aucs
    """
    ks = [3, 6, 12, 15]
    for k in ks:
        knn_name = f'KNN{k}'
        result = run_knn(k, anom)
        aucs[knn_name] = [calc_roc(*result)]
    """
    out_path = f'aucs-{drop}-fast.pkl'
    with open(out_path, 'wb') as fd:
        pickle.dump(aucs, fd)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/weather')
    parser.add_argument('--net', default='./output/weather', type=Path)
    parser.add_argument('--drop', default=0, type=float)
    parser.add_argument('--type', default='png')
    parser.add_argument('--pkl', default='aucs.pkl')
    parser.add_argument('--plot', action='store_true')
    return parser

if __name__=='__main__':
    args = make_parser().parse_args()
    if args.plot:
        plot(args.pkl, args.type)
    else:
        batch(args.data, args.net, args.drop)
