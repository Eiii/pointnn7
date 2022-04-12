from . import pointconv
from . import pointnet
from . import encodings
from . import interaction as intr
from .base import Network
from . import tpc

from functools import partial

import torch
import torch.nn as nn

HIST_FEAT_SIZE = 12
METADATA_SIZE = 19


def cross(t1, t2=None, dims=(2, 1)):
    if t2 is None:
        t2 = t1
    _t1 = t1.unsqueeze(dims[0])
    _t2 = t2.unsqueeze(dims[1])
    return _t1, _t2


def space_dist(mask, station_idx, times, allowed_neighbors, keys, points):
    # mask, station_idx, times:
    # BS x Entries
    # keys, points:
    # BS x Entries x 3
    # Calculate relative distances
    keys_e, pts_e = cross(keys, points)
    dist_vec = (keys_e - pts_e)
    sqr_dist = (dist_vec**2).sum(dim=-1)
    # Calculate valid neighbors
    times1, times2 = cross(times)
    same_time = (times1 == times2)
    sidx1, sidx2 = cross(station_idx)
    diff_stat = (sidx1 != sidx2)
    mask1, mask2 = cross(mask)
    cross_mask = mask1 * mask2
    valid = cross_mask * diff_stat * same_time
    if allowed_neighbors is not None:
        valid *= allowed_neighbors
    return valid, dist_vec, sqr_dist


def time_dist(mask, station_idx, allowed_neighbors, time_enc, keys, points):
    # mask, station_idx, times:
    # BS x Entries
    # keys, points:
    # BS x Entries x 3
    # Calculate relative distances
    keys_e, pts_e = cross(keys, points)
    dist_vec = (keys_e - pts_e).unsqueeze(-1)
    sqr_dist = (dist_vec**2).sum(dim=-1)
    # Calculate valid neighbors
    diff_time = (keys_e != pts_e)
    sidx1, sidx2 = cross(station_idx)
    same_stat = (sidx1 == sidx2)
    mask1, mask2 = cross(mask)
    cross_mask = mask1 * mask2
    valid = cross_mask * same_stat * diff_time
    if allowed_neighbors is not None:
        valid *= allowed_neighbors
    return valid, time_enc(dist_vec), sqr_dist


def target_dist(mask, times, allowed_neighbors, keys, points):
    # mask, station_idx, times:
    # BS x Entries
    # keys:
    # BS x Targets x 3
    # points:
    # BS x Entries x 3
    # Calculate relative distances
    keys_e, pts_e = cross(keys, points)
    dist_vec = (keys_e - pts_e)
    sqr_dist = (dist_vec**2).sum(dim=-1)
    # Calculate valid neighbors
    times1, times2 = cross(times)
    at_target = (times.unsqueeze(1)==0)
    mask = mask.unsqueeze(1)
    valid = mask * at_target
    if allowed_neighbors is not None:
        valid *= allowed_neighbors
    valid, _ = torch.broadcast_tensors(valid, sqr_dist)
    return valid, dist_vec, sqr_dist

class _WeatherBase(Network):
    def forward(self, hist, mask, times, station_meta, station_pos,
                station_idxs, target_pos, force_dropout=None):
        bs = hist.size(0)
        num_pts = hist.size(1)
        device = hist.device
        if force_dropout:
            dropout = force_dropout
        else:
            dropout = 0.5*torch.rand(1) if self.station_dropout == 'rand' else self.station_dropout
        if (self.training or force_dropout) and dropout > 0:
            sz = (bs, num_pts)
            prob = (1-dropout) * torch.ones(sz)
            ns_mask = torch.bernoulli(prob).bool().to(device)
            allowed = ns_mask.unsqueeze(1) * ns_mask.unsqueeze(2)
            target_allowed = ns_mask.unsqueeze(1)
        else:
            allowed = target_allowed = None
        tgt_latent = self.encode(hist, mask, times, station_meta,
                                 station_pos, station_idxs, allowed,
                                 target_allowed, target_pos)
        pred = self.pred(tgt_latent)
        return pred

    def make_predictors(self, target_size, decode_hidden, num_preds):
        args = {'in_size': target_size, 'out_size': num_preds,
                'hidden_sizes': decode_hidden, 'reduction': 'none'}
        self.pred = pointnet.SetTransform(**args)

    def get_args(self, item):
        return item['hist'], item['hist_mask'], \
               item['times'], \
               item['station_metadata'], item['station_pos'], \
               item['station_idxs'], item['target_pos']

    @staticmethod
    def _get_by_idx(idx, a):
        idxs_expand = idx.unsqueeze(-1).expand(-1, -1, a.size(-1))
        g = torch.gather(a, 1, idxs_expand)
        return g


class WeatherTPC(_WeatherBase):
    def __init__(self,
                 weight_hidden=[16, 32],
                 c_mid=32,
                 final_hidden=[256, 128, 128],
                 combine_hidden=[256, 128, 128],
                 neighborhood_sizes=[16, 16, 16],
                 latent_sizes=[32, 64, 64],
                 target_size=64,
                 decode_hidden=[64, 64, 64],
                 neighbors=8,
                 timesteps=12,
                 station_dropout=0,
                 heads=0):
        super().__init__()
        self.station_dropout = station_dropout
        # Setup
        in_size = HIST_FEAT_SIZE + METADATA_SIZE
        num_preds = 4
        self.make_encoders(in_size, weight_hidden, c_mid, final_hidden,
                           combine_hidden, neighborhood_sizes,
                           latent_sizes, neighbors, timesteps, target_size, heads)
        self.make_predictors(target_size, decode_hidden, num_preds)

    def make_encoders(self, feat_size, weight_hidden, c_mid, final_hidden,
                      combine_hidden, neighborhood_sizes,
                      latent_sizes, neighbors, timesteps, target_size, heads):
        pos_dim = 3
        self.time_encoder = encodings.PeriodEncoding(8, 20)
        self.tpc = tpc.TemporalPointConv(feat_size, weight_hidden, c_mid,
                                         final_hidden, latent_sizes, neighborhood_sizes,
                                         neighbors, timesteps, neighbors, combine_hidden,
                                         target_size, pos_dim, self.time_encoder, heads, 'space')

    def encode(self, hist, mask, times, station_meta, station_pos,
               station_idxs, allowed, target_allowed, target_pos):
        hist_pts = self._get_by_idx(station_idxs, station_pos)
        hist_meta = self._get_by_idx(station_idxs, station_meta)
        in_ = torch.cat([hist, hist_meta], dim=2)
        space_dist_fn = partial(space_dist, mask, station_idxs, times, allowed)
        time_dist_fn = partial(time_dist, mask, station_idxs, allowed, self.time_encoder.encode)
        target_dist_fn = partial(target_dist, mask, times, target_allowed)
        tgt_feats = self.tpc(in_, None, hist_pts, times, target_pos,
                             space_dist_fn, time_dist_fn, target_dist_fn)
        return tgt_feats

class WeatherGC(_WeatherBase):
    def __init__(self,
                 combine_hidden=[256, 128, 128],
                 neighborhood_sizes=[16, 16, 16],
                 latent_sizes=[32, 64, 64],
                 target_size=64,
                 decode_hidden=[64, 64, 64],
                 neighbors=8,
                 timesteps=12,
                 station_dropout=0):
        super().__init__()
        self.station_dropout = station_dropout
        # Setup
        in_size = HIST_FEAT_SIZE + METADATA_SIZE
        num_preds = 4
        self.make_encoders(in_size,
                           combine_hidden, neighborhood_sizes,
                           latent_sizes, neighbors, timesteps, target_size)
        self.make_predictors(target_size, decode_hidden, num_preds)

    def make_encoders(self, feat_size,
                      combine_hidden, neighborhood_sizes,
                      latent_sizes, neighbors, timesteps, target_size):
        pos_dim = 3
        self.time_encoder = encodings.PeriodEncoding(8, 20)
        self.tgc = tpc.TemporalGraphConv(feat_size, latent_sizes, neighborhood_sizes,
                                         combine_hidden, target_size, neighbors, timesteps,
                                         pos_dim, self.time_encoder, 'space')

    def encode(self, hist, mask, times, station_meta, station_pos,
               station_idxs, allowed, target_allowed, target_pos):
        hist_pts = self._get_by_idx(station_idxs, station_pos)
        hist_meta = self._get_by_idx(station_idxs, station_meta)
        in_ = torch.cat([hist, hist_meta], dim=2)
        space_dist_fn = partial(space_dist, mask, station_idxs, times, allowed)
        time_dist_fn = partial(time_dist, mask, station_idxs, allowed, self.time_encoder.encode)
        target_dist_fn = partial(target_dist, mask, times, target_allowed)
        tgt_feats = self.tgc(in_, None, hist_pts, times, hist_pts, target_pos,
                             space_dist_fn, time_dist_fn, target_dist_fn)
        return tgt_feats

class WeatherInteraction(_WeatherBase):
    def __init__(self,
                 edge_hidden=[16, 32],
                 combine_hidden=[256, 128, 128],
                 neighborhood_sizes=[16, 16, 16],
                 latent_sizes=[32, 64, 64],
                 target_size=64,
                 decode_hidden=[64, 64, 64],
                 neighbors=8,
                 timesteps=12,
                 station_dropout=0):
        super().__init__()
        self.station_dropout = station_dropout
        # Setup
        in_size = HIST_FEAT_SIZE + METADATA_SIZE
        num_preds = 4
        self.make_encoders(in_size, edge_hidden,
                           combine_hidden, neighborhood_sizes,
                           latent_sizes, neighbors, timesteps, target_size)
        self.make_predictors(target_size, decode_hidden, num_preds)

    def make_encoders(self, feat_size, edge_hidden,
                      combine_hidden, neighborhood_sizes,
                      latent_sizes, neighbors, timesteps, target_size):
        pos_dim = 3
        self.time_encoder = encodings.PeriodEncoding(8, 20)
        self.int = intr.TemporalInteraction(feat_size, edge_hidden,
                                            latent_sizes, neighborhood_sizes,
                                            neighbors, timesteps, combine_hidden,
                                            target_size, pos_dim, self.time_encoder, 'space')

    def encode(self, hist, mask, times, station_meta, station_pos,
               station_idxs, allowed, target_allowed, target_pos):
        hist_pts = self._get_by_idx(station_idxs, station_pos)
        hist_meta = self._get_by_idx(station_idxs, station_meta)
        in_ = torch.cat([hist, hist_meta], dim=2)
        space_dist_fn = partial(space_dist, mask, station_idxs, times, allowed)
        time_dist_fn = partial(time_dist, mask, station_idxs, allowed, self.time_encoder.encode)
        target_dist_fn = partial(target_dist, mask, times, target_allowed)
        tgt_feats = self.int(in_, None, hist_pts, times, target_pos,
                             space_dist_fn, time_dist_fn, target_dist_fn)
        return tgt_feats


class WeatherSeFT(_WeatherBase):
    def __init__(self,
                 hidden=[256, 128, 128],
                 combine_hidden=[256, 128, 128],
                 neighborhood_sizes=[16, 16, 16],
                 latent_sizes=[32, 64, 64],
                 target_size=64,
                 heads=4,
                 decode_hidden=[64, 64, 64],
                 neighbors=8,
                 timesteps=12,
                 self_attention=False,
                 station_dropout=0):
        super().__init__()
        self.station_dropout = station_dropout
        # Setup
        in_size = HIST_FEAT_SIZE + METADATA_SIZE
        num_preds = 4
        self.make_encoders(in_size, hidden, combine_hidden, neighborhood_sizes,
                           latent_sizes, neighbors, timesteps, self_attention, heads, target_size)
        self.make_predictors(target_size, decode_hidden, num_preds)

    def get_args(self, item):
        return item['hist'], item['hist_mask'], \
               item['times'], \
               item['station_metadata'], item['station_pos'], \
               item['station_idxs'], item['target_pos']

    def make_encoders(self, feat_size, hidden, combine_hidden,
                      neighborhood_sizes, latent_sizes, neighbors, timesteps,
                      self_attention, heads, target_size):
        self.time_encoder = encodings.PeriodEncoding(8, 20)
        self.space = nn.ModuleList()
        self.time = nn.ModuleList()
        self.combine = nn.ModuleList()
        in_size = feat_size
        default_args = {'hidden': hidden, 'self_attention': self_attention,
                        'heads': heads}
        for ls, n_sz in zip(latent_sizes, neighborhood_sizes):
            args = default_args.copy()
            args.update({'neighbors': neighbors, 'c_in': in_size, 'c_out': n_sz,
                         'dim': 3})
            # Space conv
            pc = pointconv.SeFT(**args)
            self.space.append(pc)
            # Station (time) conv
            args = default_args.copy()
            args.update({'neighbors': timesteps, 'c_in': in_size+n_sz,
                         'c_out': n_sz, 'dim': self.time_encoder.out_dim})
            pc = pointconv.SeFT(**args)
            self.time.append(pc)
            # Combine
            set_args = {'in_size': in_size+2*n_sz, 'out_size': ls,
                        'hidden_sizes': combine_hidden, 'reduction': 'none'}
            pn = pointnet.SetTransform(**set_args) if ls is not None else None
            self.combine.append(pn)
            in_size = ls
        args = {'hidden': hidden, 'neighbors': neighbors, 'c_in': ls,
                'dim': 3, 'c_out': target_size, 'self_attention': self_attention,
                'heads': heads}
        self.target_xf = pointconv.SeFT(**args)


"""
class WeatherMink(_WeatherBase):
    def __init__(self,
                 latent_sizes=[32, 64, 64],
                 kernel_size=21,
                 target_size=64,
                 decode_hidden=[64, 64, 64]):
        super().__init__()
        # Setup
        in_size = HIST_FEAT_SIZE + METADATA_SIZE
        num_preds = 4
        self.kernel_size = kernel_size
        self.make_encoders(in_size, latent_sizes, target_size)
        self.make_predictors(target_size, decode_hidden, num_preds)


    def get_args(self, item):
        return item['hist_quant'], item['station_pos_quant'], item['target_pos_quant'], item['target_pos']


    def forward(self, hist, station_pos, target_pos, ref_target_pos, force_dropout=None):
        target_latent = self.encode(hist, station_pos, target_pos)
        target_pred = self.pred(target_latent)
        target_size = ref_target_pos.size()[:2]
        pred_reshape = target_pred.view(*target_size, -1)
        return pred_reshape


    def encode(self, hist, station_pos, target_pos):
        sparse_in = ME.SparseTensor(hist, station_pos)
        for mink, norm, nonlin in zip(self.minks, self.norms, self.nonlins):
            sparse_out = norm(nonlin(mink(sparse_in)))
            sparse_in = sparse_out
        target_out = self.target_xf(sparse_out, coordinates=target_pos)
        return target_out.features


    def make_encoders(self, feat_size, latent_sizes, target_size):
        self.minks = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.nonlins = nn.ModuleList()
        in_size = feat_size
        kernel = ME.KernelGenerator(self.kernel_size, region_type=ME.RegionType.HYPER_CUBE, dimension=3)
        for ls in latent_sizes:
            mink = ME.MinkowskiConvolution(in_channels=in_size, out_channels=ls,
                                           dimension=3, kernel_generator=kernel)
            self.minks.append(mink)
            self.norms.append(ME.MinkowskiBatchNorm(ls))
            self.nonlins.append(ME.MinkowskiReLU())
            in_size = ls
        self.target_xf = ME.MinkowskiConvolution(in_channels=ls, out_channels=target_size,
                                                 dimension=3, kernel_generator=kernel)
"""
