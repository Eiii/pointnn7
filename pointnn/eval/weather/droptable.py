import pickle
import argparse
import torch

from collections import defaultdict


def get_net_type(net_path):
    start = net_path.rfind('/')+1
    end = net_path.index(':')
    name = net_path[start:end]
    return name


def calc_stats(losses):
    t = torch.cat(losses)
    t[t.isnan()] = float('inf')
    quants = [torch.quantile(t, x).item() for x in (0.25, 0.5, 0.75)]
    fails = (t > quants[-1]*100)
    mean = t[~fails].mean().item()
    return {'quants': quants, 'mean': mean}


def main(files):
    stats = []
    for f in files:
        print(f)
        all_losses = defaultdict(list)
        with open(f, 'rb') as fd:
            data = pickle.load(fd)
            for net_path, result in data.items():
                net_type = get_net_type(net_path)
                all_losses[net_type].append(result['loss'])
        all_stats = {k:calc_stats(v) for k,v in all_losses.items()}
        print(all_stats)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.files)
