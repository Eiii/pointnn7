import argparse

from . import common

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def get_lrs(tl):
    breakpoint()
    return [t['lr'] for t in tl]


def get_wds(tl):
    return [1-t['epoch']/0.5 for t in tl]


def plot_lr_curves(measures, out_path, skip_first, normalize):
    size = (8, 5)
    fig, ax = plt.subplots(figsize=size, dpi=200)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss Change')
    #ax.set_xscale('log')
    ax.set_yscale('log')
    all_names = list({m.name for m in measures if m.name is not None})
    print(f'Plotting {all_names}...')
    for name in all_names:
        print(f'Plotting {name}')
        # LR plot
        tls = [m._training_loss for m in measures if m.name == name]
        xs = ys = None
        for tl in tls:
            x = get_wds(tl)
            y = [t['loss'] for t in tl]
            y_array = np.array([y])
            if len(x) == 0:
                continue
            xs = np.vstack((xs, x)) if xs is not None else np.array([x])
            ys = np.vstack((ys, y)) if ys is not None else y_array
        #assert (xs == xs[0]).all()
        xs = xs[0]
        ys = np.mean(ys, axis=0)
        if skip_first:
            xs = xs[1:]
            ys = ys[1:]
        if normalize:
            ys = ys / ys[0]
            max_loss = 1.1
            ys[ys>max_loss] = max_loss
            #ys = ys - ys[0]
        calc_thresh(xs, ys)
        ax.plot(xs, ys, label=name)
    if normalize:
        ax.axhline(1, color='k')
    ax.legend()


def calc_thresh(xs, ys, amt=0.1):
    baseline = ys[0]
    best = ys.min()
    thresh = baseline + (best-baseline)*amt
    good_xs = xs[ys<thresh]
    min_xs = float(good_xs.min())
    max_xs = float(xs[ys==ys.min()])
    print(f'{min_xs:.2e} {max_xs:.2e}')


def make_plots(folder, out_path, filter, skip_first, normalize):
    measures = [r['measure'] for r in common.load_any(folder)]
    if filter:
        filters = filter.split(',')
        measures = [m for m in measures if m.name in filters]
    plot_lr_curves(measures, out_path, skip_first, normalize)
    plt.tight_layout()
    plt.savefig(out_path)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('output', nargs='?', default='out.png')
    parser.add_argument('--filter', default=None)
    parser.add_argument('--skip-first', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    make_plots(args.folder, args.output, args.filter, args.skip_first, args.normalize)

