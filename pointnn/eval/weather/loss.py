from ...data.weather import WeatherDataset, collate, collate_voxelize
from ...problem.weather import flat_loss, scaled_loss

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
        pred = model(*args).detach()
        ls = flat_loss(ds.ds, item['target'], pred)
        ls = ls.detach()
        ls = ls.view(-1, ls.size(-1))
        sls = scaled_loss(ds.ds, item['target'], pred).sum(-1)
        sls = sls.detach()
        sls = sls.view(-1)
        losses.append(ls)
        scaled_losses.append(sls)
    return torch.cat(losses), torch.cat(scaled_losses)

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

def main(data_path, net_paths, drop):
    ds = WeatherDataset(data_path, load_train=False, drop=drop, HACK_SNIP=1000)
    ds = ds.test_dataset
    print(len(ds))
    for net_path in net_paths:
        result = common.load_result(net_path)
        net = common.make_net(result)
        show_results(run_net(net, ds))

def batch(data_path, path, drop):
    names = ['TPC', 'SeFT', 'DeepSets', 'Mink']
    ds = WeatherDataset(data_path, load_train=False, drop=drop)
    ds = ds.test_dataset
    loss_dict = {}
    for name in names:
        print(name)
        net_paths = path.glob(name+'*.pkl')
        res_l = []
        scaled_l = []
        for net_path in net_paths:
            print('x')
            net = common.make_net(common.load_result(net_path))
            losses, scaled_loss = run_net(net, ds, voxel=(name=='Mink'))
            res_l.append(losses)
            scaled_l.append(scaled_loss)
        all_losses = torch.cat(res_l)
        all_scaled_losses = torch.cat(scaled_l)
        loss_dict[name] = (all_losses.cpu(), all_scaled_losses.cpu())
    with open('weather-loss.pkl', 'wb') as fd:
        pickle.dump(loss_dict, fd)

def table(path):
    with open(path, 'rb') as fd:
        data = pickle.load(fd)
    for key in data:
        all_losses, scaled_losses = data[key]
        means = all_losses.mean(dim=0)
        stds = all_losses.std(dim=0)
        count = all_losses.size(0)
        stderr = 1.96 * stds / math.sqrt(count)
        scaled_mean = scaled_losses.mean().item()
        scaled_stderr = (1.96 * scaled_losses.std() / math.sqrt(count)).item()
        strs = []
        for m, s in zip(means, stds):
            m = m.item()
            s = s.item()
            strs.append(f'$ {m:.2f} \pm {s:.2f} $')
        strs.append(f'$ {scaled_mean:.4f} \pm {scaled_stderr:.4f} $')
        print(key)
        print(' & '.join(strs))


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/weather')
    parser.add_argument('--net', default=Path('./output/weather'), type=Path)
    parser.add_argument('--drop', default=0, type=float)
    parser.add_argument('--table', action='store_true')
    return parser

if __name__=='__main__':
    args = make_parser().parse_args()
    #main(args.data, args.net, args.drop)
    if args.table:
        table(args.data)
    else:
        batch(args.data, args.net, args.drop)
