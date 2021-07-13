import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

import math
import itertools

from .. import utils


ACTIVATION_DICT= {'relu': F.relu}


def batch_linear(x, w, b):
    return x.matmul(w.transpose(-1, -2)) + b

class _Base(nn.Module):
    def __init__(self, block_size, layer_blocks, shape_size, in_size, out_size,
                 batchnorm, activation):
        super().__init__()
        self.block_size = block_size
        self.layer_blocks = layer_blocks
        self.shape_size = shape_size
        self.in_size = in_size
        self.out_size = out_size
        self.batchnorm = batchnorm
        self.activation = ACTIVATION_DICT[activation]

    def _forward(self, x, weights, biases, norms):
        if norms is not None:
            ext_norms = itertools.chain(norms, (None,))
            params = list(zip(weights, biases, ext_norms))
        else:
            blank_norms = [None]*len(weights)
            params = list(zip(weights, biases, blank_norms))
        for w, b, norm in params[:-1]:
            x = batch_linear(x, w, b)
            x = self.activation(x)
            if norm is not None:
                x_perm = x.permute(0, 2, 1)
                x_perm = norm(x_perm)
                x = x_perm.permute(0, 2, 1)
        w, b, norm = params[-1]
        assert norm is None
        x = batch_linear(x, w, b)
        return x

    def _create_batchnorm(self, input_size):
        self.norms = nn.ModuleList()
        actual_sizes = [input_size] + \
            [self.block_size*x for x in self.layer_blocks] + [self.out_size]
        prev_norm = None
        for in_sz, out_sz in utils.pairwise(actual_sizes):
            if prev_norm:
                self.norms.append(prev_norm)
            prev_norm = nn.BatchNorm1d(out_sz, momentum=0.01)

class PointNet(_Base):
    def __init__(self, block_size, layer_blocks, shape_size, in_size,
                 out_size, batchnorm=True, activation='relu' ):
        super().__init__(block_size, layer_blocks, shape_size, in_size,
                         out_size, batchnorm, activation)
        self._create_params()
        self._init_params()

    def _create_params(self):
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        actual_sizes = [self.shape_size+self.in_size] + \
            [self.block_size*x for x in self.layer_blocks] + [self.out_size]
        for in_sz, out_sz in utils.pairwise(actual_sizes):
            weight = Parameter(torch.zeros(out_sz, in_sz))
            bias = Parameter(torch.zeros(out_sz))
            self.weights.append(weight)
            self.biases.append(bias)
        if self.batchnorm:
            self._create_batchnorm(self.shape_size+self.in_size)

    def _init_params(self):
        for w, b in zip(self.weights, self.biases):
            nn.init.kaiming_uniform_(w)
            bound = 1/math.sqrt(w.size(1))
            nn.init.uniform_(b, -bound, bound)

    def set_latent(self, latent):
        self.latent = latent

    def forward(self, x):
        repeat_lat = self.latent.unsqueeze(1).repeat(1, x.size(1), 1)
        combined = torch.cat([repeat_lat, x], dim=2)
        norms = self.norms if self.batchnorm else None
        return self._forward(combined, self.weights, self.biases, norms)

class HyperPointNet(_Base):
    def __init__(self, block_size, layer_blocks, shape_size, in_size,
                 out_size, embedding_size, hyper_hidden, z_layers,
                 batchnorm=True, activation='relu'):
        super().__init__(block_size, layer_blocks, shape_size, in_size,
                         out_size, batchnorm, activation)
        self.embedding_size = embedding_size
        self.hyper_hidden = hyper_hidden
        self.z_layers = z_layers
        self.total_blocks = self._calc_blocks()
        self._create_params()
        self._init_params()

    def _calc_blocks(self):
        total_blocks = sum(in_*out for in_, out in utils.pairwise(self.layer_blocks))
        return total_blocks

    def _make_z_net(self):
        layers = []
        sizes = [self.shape_size] + self.z_layers + [self.total_blocks*self.embedding_size]
        for in_, out in utils.pairwise(sizes):
            layers.append(nn.Linear(in_, out))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out))
        self.z_net = nn.Sequential(*layers)

    def _create_params(self):
        # Global latent -> Block Embedding
        # BS x global latent size -> BS x Num blocks x Block embedding size
        self._make_z_net()
        # Embedding -> a (tb*hyper_hidden)
        # BS x Num blocks x Block embedding -> BS x Num blocks x N_out x hidden
        self.s2 = nn.Linear(self.embedding_size, self.block_size*self.hyper_hidden)
        # BS x Num blocks x N_out x hidden -> BS x Num blocks x N_out x  (N_in+1)
        w_out = torch.zeros(self.hyper_hidden, self.block_size+1)
        b_out = torch.zeros(self.block_size+1)
        self.w_out = Parameter(w_out)
        self.b_out = Parameter(b_out)
        # First & Last weights
        first_size = self.block_size*self.layer_blocks[0]
        self.first_weight = Parameter(torch.zeros(first_size, self.in_size))
        self.first_bias = Parameter(torch.zeros(first_size))
        last_size = self.block_size*self.layer_blocks[-1]
        self.last_weight = Parameter(torch.zeros(self.out_size, last_size))
        self.last_bias = Parameter(torch.zeros(self.out_size))
        # Batch norm
        if self.batchnorm:
            self._create_batchnorm(self.in_size)

    def _init_params(self):
        pairs = [(self.w_out, self.b_out),
                 (self.first_weight, self.first_bias),
                 (self.last_weight, self.last_bias)]
        for weight, bias in pairs:
            nn.init.kaiming_uniform_(weight)
            bound = 1/math.sqrt(weight.size(1))
            nn.init.uniform_(bias, -bound, bound)

    def _combine_final(self, as_):
        w_expand = self.w_out.view(1, 1, 1, *self.w_out.size())
        b_expand = self.b_out.view(1, 1, 1, *self.b_out.size())
        as_expand = as_.unsqueeze(-1)
        return (w_expand*as_expand).sum(dim=-2) + b_expand

    def _combine_blocks(self, blocks):
        block_it = iter(blocks.unbind(1))
        out_weights = []
        out_biases = []
        for in_, out in utils.pairwise(self.layer_blocks):
            ll = [[next(block_it) for _ in range(out)] for _ in range(in_)]
            combined = torch.cat([torch.cat(bs, dim=1) for bs in ll], dim=2)
            weights = combined[:, :, :-in_]
            biases = combined[:, :, -1:].view(weights.size(0), 1, -1)
            out_weights.append(weights)
            out_biases.append(biases)
        return out_weights, out_biases

    def set_latent(self, latent):
        flat_zs = self.z_net(latent)
        zs = flat_zs.view(flat_zs.size(0), self.total_blocks, self.hyper_hidden)
        flat_as = self.s2(zs)
        as_ = flat_as.view(flat_as.size(0), flat_as.size(1), self.block_size, self.hyper_hidden)
        blocks = self._combine_final(as_)
        self.hyper_weights, self.hyper_biases = self._combine_blocks(blocks)

    def forward(self, x):
        weights = [self.first_weight] + self.hyper_weights + [self.last_weight]
        biases = [self.first_bias] + self.hyper_biases + [self.last_bias]
        norms = self.norms if self.batchnorm else None
        return self._forward(x, weights, biases, norms)


if __name__ == '__main__':
    bs = 5
    num_pts = 16
    lat_sz = 16
    noise_sz = 8
    args = [32, [1, 2, 2, 1], lat_sz, noise_sz, 3]
    hp = PointNet(*args)
    print(hp)
    x = torch.randn(bs, num_pts, noise_sz)
    lat = torch.zeros(bs, num_pts)
    hp.set_latent(lat)
    y = hp(x)
    hyper_hidden = 8
    hp = HyperPointNet(*args, hyper_hidden)
    print(hp)
    hp.set_latent(lat)
    y = hp(x)
