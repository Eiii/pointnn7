from ...data.traffic import METRDataset, collate
from ...nets.traffic.baseline import Random, Nearest, Mean
from .. import common
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from matplotlib import collections as coll
from matplotlib.animation import ArtistAnimation

from pathlib import Path
import argparse
import torch

def to_cuda(d):
    return {k:v.cuda() if v is not None else None for k,v in d.items()}

def make_parser():
    def_data = './data/traffic/METR-LA/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=Path(def_data))
    parser.add_argument('--net', nargs='+')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--spectral', action='store_true')
    return parser

def plot_traffic(ax, ds, sensor, norm_info):
    first_x = None
    mean, std = norm_info
    xs = []
    ys = []
    for entry in ds.entries:
        ref = entry.center
        first_x = first_x or ref
        time = ds.raw_times[ref]
        col = ds.raw_df.loc[ref]
        if type(sensor) == int:
            sensor = str(sensor)
        val = col[str(sensor)]
        val = val*std + mean
        xs.append(ref-first_x)
        ys.append(val)
    ax.plot(xs, ys, label='actual', c='k')

def plot_preds(ax, ds, preds, sensor_id, norm_info, name='pred'):
    mean, std = norm_info
    xs = range(len(preds))
    ys = []
    for item, p in zip(ds, preds):
        x = collate([item])
        mask = x['tgt_id'] == sensor_id
        val = p[mask].item()
        val = val*std + mean
        ys.append(val)
    ax.plot(xs, ys, label=name)
    ax.set_ylim([0, 70])

@torch.no_grad()
def predict(net, ds):
    preds = []
    for item in ds:
        x = collate([item])
        x = to_cuda(x)
        args = net.get_args(x)
        pred = net(*args).cpu()
        preds.append(pred)
    return preds

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

def single(base, net_paths, norm, spec):
    valid_fn = lambda date: date.week % 10 == 0 and date.month == 5 and date.day == 20
    dims = 50
    ds = METRDataset(base, valid_fn, norm, spec, dims)
    norm_info = ds.norm_info if norm else [0, 1]
    all_ids = ds.raw_df.columns.tolist()
    assert len(ds) == 288
    models = [make_net(path) for path in net_paths]
    preds = [predict(model, ds) for model in models]
    names = ['TPC', 'Spectral', 'Blank']
    mean_model = make_baseline('mean')
    mean_preds = predict(mean_model, ds)
    for sensor_id in all_ids:
        fig, ax = plt.subplots(figsize=(10,10), dpi=100)
        print(sensor_id)
        plot_traffic(ax, ds, sensor_id, norm_info)
        sensor_idx = ds.sensor_id_to_idx[int(sensor_id)]
        for pred, name in zip(preds, names):
            plot_preds(ax, ds, pred, sensor_idx, norm_info, name)
        plot_preds(ax, ds, mean_preds, sensor_idx, norm_info, 'mean')
        ax.legend()
        fig.savefig(f'day/{sensor_id}.png')
        plt.close(fig)

if __name__ == '__main__':
    args = make_parser().parse_args()
    single(args.data, args.net, args.normalize, args.spectral)
