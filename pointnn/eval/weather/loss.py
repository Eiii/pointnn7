from ...data.weather import WeatherDataset, collate
from ...problem.weather import flat_loss, scaled_loss
from .. import common
from torch.utils.data import DataLoader
import math
import argparse
import torch
import pickle


def seq_to_device(d, device):
    if isinstance(d, dict):
        return {k: v.to(device) for k, v in d.items()}
    elif isinstance(d, list):
        return [v.to(device) for v in d]


def pred_safe(model, ds, bs, device):
    while bs != 1:
        print(f'batch size={bs}')
        try:
            return pred(model, ds, bs, device)
        except RuntimeError as e:
            print(e)
            bs = bs // 2
    raise RuntimeError()


def pred(model, ds, bs, device):
    losses = []
    scaled_losses = []
    loader = DataLoader(ds, batch_size=bs, collate_fn=collate, num_workers=4)
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


def run_net(net, ds, bs):
    with torch.no_grad():
        net = net.eval().cuda()
        #ens = SelfEnsemble(net, 3, {'force_dropout': 0.2})
        return pred_safe(net, ds, bs, device='cuda')


def show_results(losses):
    mean_preds = losses.mean(dim=0)
    print(mean_preds.tolist())
    mean = mean_preds.mean()
    print(mean.tolist())
    return mean


def batch(data_path, net_paths, bs, drop, out_path):
    ds = WeatherDataset(data_path, load_train=False, drop=drop)
    ds = ds.test_dataset
    loss_dict = {}
    for net_path in net_paths:
        print(net_path)
        net = common.make_net(common.load_result(net_path))
        loss, scaled_loss = run_net(net, ds, bs)
        loss_dict[net_path] = {'loss': scaled_loss, 'mse': loss}
        with open(out_path, 'wb') as fd:
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
    parser.add_argument('--nets', nargs='+')
    parser.add_argument('--out', default='weather-loss.pkl')
    parser.add_argument('--drop', default=0, type=float)
    parser.add_argument('--bs', default=32, type=int)
    parser.add_argument('--table', action='store_true')
    return parser

if __name__=='__main__':
    args = make_parser().parse_args()
    batch(args.data, args.nets, args.bs, args.drop, args.out)
