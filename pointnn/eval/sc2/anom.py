from ...data.starcraft import StarcraftDataset, parse_frame
from .. import common
from ...problem import sc2_loss
import argparse
import torch
import numpy as np
import pickle

from pathlib import Path
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import sem
import sklearn.metrics

MAX_UNITS=60

def pred(scene, net):
    in_ = scene['pre'].unsqueeze(0)
    hist = scene['hist'].unsqueeze(0)
    mask = scene['mask'].unsqueeze(0)
    p = net(in_, mask)
    return p

def loss(scene, pred):
    scene['pre'] = scene['pre'].unsqueeze(0)
    scene['post'] = scene['post'].unsqueeze(0)
    return sc2_loss(scene, pred)

def thresh_calc(net):
    data_path = 'data/zerganom'
    ds = StarcraftDataset(data_path, device='cpu', max_units=MAX_UNITS,
                          hist_len=1, hist_skip=1, hide_type=True,
                          include_anom=True)
    losses = []
    anoms = []
    num_ep = ds.num_episodes()
    for count, ep in enumerate(ds.episode_frames(e) for e in range(num_ep)):
        print(f'{count}/{num_ep}')
        key_losses = defaultdict(list)
        for item in ep:
            l = loss(item, pred(item, net)).sum(dim=0).sum(dim=0)
            mask_idxs = item['mask'].nonzero()[:, 0]
            ll = l.tolist()[mask_idxs]
            ts = item['tag'].tolist()[mask_idxs]
            for l, t in zip(ll, ts):
                key_losses[t].append(l)
        anom_tags = ep[0]['tag'][ep[0]['anom_idxs']].tolist()
        all_keys = ep[0]['tag'][:ep[0]['mask'].sum()].tolist()
        ml = [np.mean(key_losses[k])+1.96*sem(key_losses[k]) for k in all_keys]
        losses += ml
        anoms += [1 if k in anom_tags else 0 for k in all_keys]
    return np.array(losses), np.array(anoms)

def thresh_calc_scene(net):
    data_path = 'data/zerglotsboth'
    ds = StarcraftDataset(data_path, device='cpu', max_units=MAX_UNITS,
                          hist_len=1, hist_skip=1, hide_type=True,
                          include_anom=True)
    losses = []
    anoms = []
    num_ep = ds.num_episodes()
    for count, ep in enumerate(ds.episode_frames(e) for e in range(num_ep)):
        print(f'{count}/{num_ep}')
        all_losses = list()
        for item in ep:
            l = loss(item, pred(item, net)).sum(dim=0).sum(dim=0)
            mask_idxs = item['mask'].nonzero()[:, 0]
            ll = l.tolist()[mask_idxs]
            all_losses += ll
        has_anom = len(ep[0]['anom_idxs']) > 0
        losses += [np.mean(all_losses)+1.96*sem(all_losses)]
        anoms += [1] if has_anom else [0]
    return np.array(losses), np.array(anoms)

def thresh_apply(losses, anoms, t):
    flagged = np.where(losses > t)
    notflagged = np.where(losses < t)
    real = np.where(anoms == 1)
    notreal = np.where(anoms != 1)
    tp = len(np.intersect1d(flagged, real))
    fp = len(np.setdiff1d(flagged, real))
    tn = len(np.intersect1d(notflagged, notreal))
    fn = len(np.setdiff1d(notflagged, notreal))
    tpr = tp / (tp+fn)
    fpr = fp / (tn+fp)
    return tpr, fpr

def plot_roc(data):
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([0,1],[0,1], alpha=0.25, c='black')
    tpr, fpr = zip(*data)
    ax.plot(fpr, tpr)
    auc = sklearn.metrics.auc(fpr, tpr)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'Large scene. ROC, CI threshold. AUC={auc:.4f}')
    fig.savefig('roc_lots_ci.png')

def gen_roc(losses, anoms):
    c = len(losses)//100
    sort_loss = np.sort(losses)[::c]
    pts = []
    for x in sort_loss:
        result = thresh_apply(losses, anoms, x)
        pts.append(result)
    plot_roc(pts)

def plot_hists(losses, anoms):
    pairs = list(zip(anoms, losses))
    norm = [l for a, l in pairs if a == 0]
    anom = [l for a, l in pairs if a == 1]
    range_ = min(norm+anom), max(norm+anom)
    bins = 50
    norm_hist = np.histogram(norm, bins, range_)[0]
    anom_hist = np.histogram(anom, bins, range_)[0]
    xs = np.arange(bins)
    width = 0.5
    fig, ax = plt.subplots()
    r1 = ax.bar(xs+width/2, norm_hist, width, label='Normal')
    r2 = ax.bar(xs-width/2, anom_hist, width, label='Fast')
    ax.legend()
    fig.savefig('thresh_dist.png')

def main(net_path):
    net_path = net_path.glob('*.net')
    net_path = list(net_path)[0]
    measure = common.load_measure(net_path.with_suffix('.pkl'))
    net = common.load_net(net_path.with_suffix('.net'), measure)
    net = net.eval()
    with open('dump.pkl', 'wb') as fd:
        losses, anoms = thresh_calc_scene(net)
        pickle.dump((losses, anoms), fd)
    gen_roc(losses, anoms)
    plot_hists(losses, anoms)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=Path)
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.net)
