from argparse import ArgumentParser
from ...problem.traffic import TrafficMETR, element_loss, collate
from ...data.traffic import collate
from ...nets.traffic.baseline import Random, Nearest, Mean
from .. import common

import torch
from torch.utils.data import DataLoader

def to_cuda(d):
    return {k:v.cuda() for k,v in d.items()}

def main(data_path, net_path, norm):
    prob = TrafficMETR(data_path, norm)
    ds = prob.valid_dataset
    net = make_net(net_path)
    # net = make_baseline('mean')
    eval_model(ds, net, norm)

@torch.no_grad()
def eval_model(ds, net, norm):
    losses = []
    norm_info = ds.norm_info if norm else None
    loader = DataLoader(ds, batch_size=10, collate_fn=collate)
    for item in loader:
        item = to_cuda(item)
        args = net.get_args(item)
        pred = net(*args)
        err = element_loss(item, pred, norm_info).cpu()
        losses.append(err)
    losses = torch.cat(losses)
    losses = losses ** 0.5
    print(losses.mean())

def make_net(path):
    net = common.make_net(common.load_result(path))
    net = net.cuda().eval()
    return net

def make_baseline(type_):
    if type_ == 'random':
        return Random([0, 70])
    elif type_ == 'nearest':
        return Nearest()
    elif type_ == 'mean':
        return Mean()

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--data', default='data/traffic/METR-LA')
    parser.add_argument('--net')
    parser.add_argument('--normalize', action='store_true')
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.data, args.net, args.normalize)
