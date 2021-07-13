from ...data.starcraft import StarcraftDataset, parse_frame, collate, collate_voxelize
from ...problem.starcraft import sc2_frame_loss

from .. import common

import math
import functools
import argparse
import itertools
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pickle

def seq_to_device(d, device):
    if isinstance(d, dict):
        return {k:v.to(device) for k,v in d.items()}
    elif isinstance(d, list):
        return [v.to(device) for v in d]

def pred(model, ds, voxel=False, device='cpu'):
    if voxel:
        col = functools.partial(collate_voxelize, 6000)
    else:
        col = collate
    losses = []
    scaled_losses = []
    loader = DataLoader(ds, batch_size=64, collate_fn=col, num_workers=8)
    for item in loader:
        item = seq_to_device(item, device)
        args = model.get_args(item)
        pred = model(*args)
        ls = sc2_frame_loss(item, pred, reduction='none')
        ls = ls.detach()
        losses.append(ls)
    return torch.cat(losses)

def run_net(net, ds, voxel):
    with torch.no_grad():
        net = net.eval().cuda()
        #ens = SelfEnsemble(net, 3, {'force_dropout': 0.2})
        return pred(net, ds, voxel, device='cuda')

def show_results(losses):
    mean_preds = losses.mean(dim=0)
    print(mean_preds.tolist())
    mean = mean_preds.mean()
    print(mean.tolist())
    return mean

def batch(data_path, path):
    names = ['SeFT', 'DeepSets', 'Mink', 'TPC']
    ds = StarcraftDataset(data_path, num_pred=4, max_hist=5, hist_dist='uniform',
                          hist_dist_args={'max': 10}, pred_dist='fixed',
                          pred_dist_args={'ts': [1, 2, 4, 7]},
                          frame_skip=5)
    loss_dict = {}
    for name in names:
        print(name)
        net_paths = path.glob(name+'*.pkl')
        res_l = []
        for net_path in net_paths:
            print('x')
            net = common.make_net(common.load_result(net_path))
            losses = run_net(net, ds, voxel=(name=='Mink'))
            res_l.append(losses)
        all_losses = torch.cat(res_l)
        loss_dict[name] = all_losses.cpu()
    with open('sc-loss.pkl', 'wb') as fd:
        pickle.dump(loss_dict, fd)

def table(path):
    with open(path, 'rb') as fd:
        data = pickle.load(fd)
    for key in data:
        all_losses = data[key]
        means = all_losses.mean(dim=0)
        stds = all_losses.std(dim=0)
        count = all_losses.size(0)
        stderr = 1.96 * stds / math.sqrt(count)
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


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/sc2scene')
    parser.add_argument('--net', default=Path('./output/sc2demo'), type=Path)
    parser.add_argument('--table', action='store_true')
    return parser

if __name__=='__main__':
    args = make_parser().parse_args()
    if args.table:
        table(args.data)
    else:
        batch(args.data, args.net)
