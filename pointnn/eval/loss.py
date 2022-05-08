import pickle
import argparse
import torch
from collections import defaultdict


def main(weather_path, sc2_path, traffic_path, out_file):
    def _load(p):
        if p is None:
            return None
        with open(p, 'rb') as fd:
            return pickle.load(fd)
    weather = _load(weather_path)
    sc2 = _load(sc2_path)
    traffic = _load(traffic_path)
    weather_stats = group_results(weather)
    traffic_stats = group_results(traffic)
    sc2_stats = group_results(sc2)
    results = {'sc2': sc2_stats, 'weather': weather_stats, 'traffic': traffic_stats}
    with open(out_file, 'wb') as fd:
        pickle.dump(results, fd)


def make_table(weather, traffic, sc2):
    pass


def get_net_type(net_path):
    start = net_path.rfind('/')+1
    end = net_path.index(':')
    return net_path[start:end]


def calc_stats(losses):
    t = torch.cat(losses)
    t[t.isnan()] = float('inf')
    quants = [torch.quantile(t, x).item() for x in (0.25, 0.5, 0.75)]
    fails = (t > t.median()*1000)
    fail_pct = (fails).sum().item() / len(t)
    mean = t[~fails].mean().item()
    return {'quants': quants, 'fails': fail_pct, 'mean': mean}


def group_results(result_dict):
    if not result_dict:
        return None
    all_losses = defaultdict(list)
    for net_path, result in result_dict.items():
        net_type = get_net_type(net_path)
        # SC2 hack...
        if 'loss' not in result and 'losses' in result:
          losses = result['losses'].sum(dim=-1)
        else:
          losses = result['loss']
        all_losses[net_type].append(losses)
    all_stats = {k: calc_stats(v) for k, v in all_losses.items()}
    return all_stats


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weather', default=None)
    parser.add_argument('--sc2', default=None)
    parser.add_argument('--traffic', default=None)
    parser.add_argument('--out', default='loss-stats.pkl')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.weather, args.sc2, args.traffic, args.out)
