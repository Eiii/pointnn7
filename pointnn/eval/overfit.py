import argparse

from . import common

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def calc_thresh(x, tl):
    prev_t = 0
    prev_loss = 0
    if x < tl[0]['epoch']:
        return 9999
    for t in tl:
        e = t['epoch']
        if prev_t < x <= e:
            return prev_loss
        prev_t = e
        prev_loss = t['loss']
    return tl[-1]['loss']

def plot_training_curves(measures, out_path, show_training=False):
    size = (8, 5)
    fig, ax = plt.subplots(figsize=size, dpi=200)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Overfit %')
    all_names = list({m.name for m in measures if m.name is not None})
    print(f'Plotting {all_names}...')
    for name in all_names:
        print(f'Plotting {name}')
        # Valid loss
        xs = []
        ys = []
        ts = []
        for vl, tl in [(m._valid_stats, m._training_loss) for m in measures if m.name == name]:
            x = [v['epoch'] for v in vl]
            y = [v['loss'] for v in vl]
            thresh = [2*calc_thresh(_x, tl) for _x in x]
            t = vl[-1]['time']
            y_array = np.array(y)
            y_array = (y_array > thresh).astype(float)
            print(len(x))
            xs.append(x)
            ys.append(y_array)
            ts.append(t)
        max_len = max(len(x) for x in xs)
        xs = np.stack([x for x in xs if len(x) == max_len])
        ys = np.stack([y for y in ys if len(y) == max_len])
        ts = [t for t in ts if len(x) == max_len]
        xs = xs.mean(0)
        ys = np.mean(ys, axis=0)
        print(name)
        main_plot = ax.plot(xs, ys, label=name)
        main_color = main_plot[-1].get_color()

    ax.legend()


def make_plots(folder, out_path, filter=None, training=False):
    measures = [r['measure'] for r in common.load_any(folder)]
    if filter:
        filters = filter.split(',')
        measures = [m for m in measures
                    if any(f in m.name for f in filters)]
    plot_training_curves(measures, out_path, training)
    plt.tight_layout()
    plt.savefig(out_path)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--output', default='out.png')
    parser.add_argument('--filter', default=None)
    parser.add_argument('--training', action='store_true')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    make_plots(args.folder, args.output, args.filter, args.training)

