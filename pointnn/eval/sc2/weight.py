from .. import common
import argparse
import torch
import matplotlib.pyplot as plt


def plot_weights(pos, weights, out='out.png'):
    fig, ax = plt.subplots()
    xs, ys = pos
    ax.pcolormesh(xs, ys, weights)
    fig.savefig(out)
    plt.close(fig)


def calc_weights_tpc(net, layer=0):
    steps = 150
    xs = torch.linspace(-1, 1, steps=steps)
    ys = torch.linspace(-1, 1, steps=steps)
    weight_grid = torch.cartesian_prod(xs, ys)
    wc = net.tpc.space_convs[layer].weight_conv
    weights = wc(weight_grid).detach()
    weights = weights.reshape(xs.size(0), ys.size(0), -1)
    return (xs, ys), weights


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net')
    return parser


def main(net_path):
    net = common.make_net(common.load_result(net_path)).eval()
    pos, weights = calc_weights_tpc(net)
    for i in range(16):
        w = weights[:, :, i]
        plot_weights(pos, w, f'out{i}.png')


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.net)
