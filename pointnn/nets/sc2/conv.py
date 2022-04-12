from . import dists
from .. import tpc
from .. import interaction as intr
from .. import pointconv
from .. import encodings
from ..base import Network
from ...data.starcraft import parse_frame
from ..pointnet import SetTransform

import torch
import torch.nn as nn
import torch.nn.functional as F

#import MinkowskiEngine as ME

from functools import partial
from collections import defaultdict

class _SC2Common(Network):
    def make_decoders(self, in_size, decode_hidden):
        h_args = {'in_size': in_size, 'out_size': 1,
                  'hidden_sizes': decode_hidden, 'reduction': 'none'}
        o_args = {'in_size': in_size, 'out_size': 7,
                  'hidden_sizes': decode_hidden, 'reduction': 'none'}
        p_args = {'in_size': in_size, 'out_size': 2,
                  'hidden_sizes': decode_hidden, 'reduction': 'none'}
        a_args = {'in_size': in_size, 'out_size': 2,
                  'hidden_sizes': decode_hidden, 'reduction': 'none'}
        self.health_net = SetTransform(**h_args)
        self.shield_net = SetTransform(**h_args)
        self.ori_net = SetTransform(**o_args)
        self.pos_net = SetTransform(**p_args)
        self.alive_net = SetTransform(**a_args)

    def get_args(self, item):
        keys = ('data', 'ids', 'ts', 'mask', 'pred_ids', 'pred_ts')
        return [item[k] for k in keys]

    def forward(self, data, ids, ts, mask, pred_ids, pred_ts):
        target_feats = self.encode_targets(data, ids, ts, mask, pred_ids, pred_ts)
        preds = self.predict(target_feats)
        return self.assemble_frame(*preds)

    def encode_targets(self, data, ids, ts, mask, pred_ids, pred_ts):
        enc_data = self.encode(data, ids, ts, mask)
        target_feats = self.calc_targets(enc_data, ids, ts, mask, pred_ids, pred_ts)
        return target_feats

    def predict(self, unit_feats):
        healths = self.health_net(unit_feats)
        shields = self.shield_net(unit_feats)
        oris = self.ori_net(unit_feats)
        poss = self.pos_net(unit_feats)
        alive = self.alive_net(unit_feats)
        return [healths, shields, oris, poss, alive]

    def assemble_frame(self, health, shield, ori, pos, alive):
        return {'health': health,
                'shields': shield,
                'ori': ori,
                'pos': pos,
                'alive': alive}


class SC2SeFT(_SC2Common):
    def __init__(self,
                 neighborhood_sizes,
                 latent_sizes,
                 target_size,
                 combine_hidden,
                 hidden,
                 decode_hidden,
                 neighbors,
                 timesteps,
                 self_attention=False,
                 ):
        super().__init__()
        feat_size = 16
        self.make_encoders(feat_size, neighborhood_sizes, latent_sizes, target_size,
                           neighbors, combine_hidden, timesteps, hidden, self_attention)
        self.make_decoders(target_size, decode_hidden)

    def make_encoders(self, feat_size, neighborhood_sizes, latent_sizes, target_size,
                      neighbors, combine_hidden, timesteps, hidden, self_attention):
        self.time_encoder = encodings.PeriodEncoding(8, 10)
        self.space_sets = nn.ModuleList()
        self.time_sets = nn.ModuleList()
        self.combine_mlps = nn.ModuleList()
        in_size = feat_size
        default_args = {'hidden': hidden, 'self_attention': self_attention,
                        'heads': 4}
        assert len(latent_sizes) == len(neighborhood_sizes)
        for ls, n_sz in zip(latent_sizes, neighborhood_sizes):
            # Space neighborhood
            args = default_args.copy()
            args.update({'neighbors': neighbors, 'c_in': in_size, 'c_out': n_sz,
                         'dim': 2})
            pc = pointconv.SeFT(**args)
            self.space_sets.append(pc)
            # Time neighborhood
            args = default_args.copy()
            args.update({'neighbors': timesteps, 'c_in': in_size+n_sz,
                         'c_out': n_sz, 'dim': self.time_encoder.out_dim})
            pc = pointconv.SeFT(**args)
            self.time_sets.append(pc)
            # MLP to next layer
            mlp_args = {'in_size': in_size+2*n_sz, 'out_size': ls,
                        'hidden_sizes': combine_hidden, 'reduction': 'none'}
            pn = SetTransform(**mlp_args)
            self.combine_mlps.append(pn)
            in_size = ls
        # Target set
        args = {'hidden': [x*2 for x in hidden],
                'neighbors': timesteps, 'c_in': ls, 'c_out': target_size,
                'dim': self.time_encoder.out_dim, 'self_attention': self_attention,
                'heads': 4}
        self.target_set = pointconv.SeFT(**args)

    def encode(self, data, ids, ts, mask):
        pf = parse_frame(data)
        # Add time to unit positions
        pos = pf['pos']
        # Initial setup to execute network
        # Spatial convolution operates on XYZ+T of units
        space_points = pos
        space_dist_fn = partial(dists.space, mask, ts)
        # Time convolution
        time_points = ts
        time_dist_fn = partial(dists.time, mask, ids, self.time_encoder.encode)
        space_in = data
        for space, time, comb in \
                zip(self.space_sets, self.time_sets, self.combine_mlps):
            # Calculate spatial convolution
            space_nei = space(space_points, space_points, space_in, space_dist_fn)
            # Combine input+output feats of space conv as input for time conv
            time_in = torch.cat([space_in, space_nei], dim=2)
            # Run time convolution
            time_nei = time(time_points, time_points, time_in, time_dist_fn)
            combined = torch.cat([space_in, space_nei, time_nei], dim=2)
            # Construct input to next space conv by appending time conv
            # output
            space_in = comb(combined)
        return space_in

    def calc_targets(self, data, ids, ts, mask, pred_ids, pred_ts):
        expand_pred_ts = pred_ts.unsqueeze(-1)
        expand_pred_ids = pred_ids.unsqueeze(-2)
        expand_pred_ts, expand_pred_ids = torch.broadcast_tensors(expand_pred_ts, expand_pred_ids)
        flat_pred_ts = expand_pred_ts.reshape(expand_pred_ts.size(0), -1)
        flat_pred_ids = expand_pred_ids.reshape(expand_pred_ids.size(0), -1)
        target_dist_fn = partial(dists.target, mask, flat_pred_ids, ids, self.time_encoder.encode)
        target_feats = self.target_set(flat_pred_ts, ts, data, target_dist_fn)
        # Restore dimensions
        target_feats = target_feats.view(expand_pred_ts.size()+(target_feats.size(-1),))
        return target_feats


class SC2TPC(_SC2Common):
    def __init__(self,
                 neighborhood_sizes,
                 latent_sizes,
                 target_size,
                 combine_hidden,
                 weight_hidden,
                 c_mid,
                 final_hidden,
                 decode_hidden,
                 neighbors, timesteps, heads=0):
        super().__init__()
        feat_size = 16
        pos_dim = 2
        self.time_encoder = encodings.PeriodEncoding(8, 10)
        self.make_tpc(feat_size, weight_hidden, c_mid, final_hidden,
                      latent_sizes, neighborhood_sizes, neighbors,
                      timesteps, combine_hidden, target_size, pos_dim,
                      self.time_encoder, heads)
        self.make_decoders(target_size, decode_hidden)

    def make_tpc(self, feat_size, weight_hidden, c_mid, final_hidden,
                 latent_sizes, neighborhood_sizes, neighbors,
                 timesteps, combine_hidden, target_size, pos_dim,
                 time_encoder, heads):
        self.tpc = tpc.TemporalPointConv(feat_size, weight_hidden, c_mid, final_hidden,
                                         latent_sizes, neighborhood_sizes, neighbors,
                                         timesteps, timesteps, combine_hidden,
                                         target_size, pos_dim, time_encoder, heads)

    def run_tpc(self, data, ids, pos, ts, flat_pred_ts, space_dist_fn, time_dist_fn, target_dist_fn):
        out = self.tpc(data, ids, pos, ts, flat_pred_ts, space_dist_fn, time_dist_fn, target_dist_fn)
        return out

    def encode_targets(self, data, ids, ts, mask, pred_ids, pred_ts):
        pf = parse_frame(data)
        pos = pf['pos']
        space_dist_fn = partial(dists.space, mask, ts)
        time_dist_fn = partial(dists.time, mask, ids, self.time_encoder.encode)
        # Target info
        expand_pred_ts = pred_ts.unsqueeze(-1)
        expand_pred_ids = pred_ids.unsqueeze(-2)
        expand_pred_ts, expand_pred_ids = torch.broadcast_tensors(expand_pred_ts, expand_pred_ids)
        flat_pred_ts = expand_pred_ts.reshape(expand_pred_ts.size(0), -1)
        flat_pred_ids = expand_pred_ids.reshape(expand_pred_ids.size(0), -1)
        target_dist_fn = partial(dists.target, mask, flat_pred_ids, ids, self.time_encoder.encode)
        out = self.run_tpc(data, ids, pos, ts, flat_pred_ts, space_dist_fn, time_dist_fn, target_dist_fn)
        target_feats = out.view(expand_pred_ts.size()+(out.size(-1),))
        return target_feats


class SC2Interaction(_SC2Common):
    def __init__(self,
                 neighborhood_sizes,
                 latent_sizes,
                 target_size,
                 combine_hidden,
                 edge_hidden,
                 decode_hidden,
                 neighbors, timesteps):
        super().__init__()
        feat_size = 16
        pos_dim = 2
        self.time_encoder = encodings.PeriodEncoding(8, 10)
        self.make_xxx(feat_size, edge_hidden,
                      latent_sizes, neighborhood_sizes, neighbors,
                      timesteps, combine_hidden, target_size, pos_dim,
                      self.time_encoder)
        self.make_decoders(target_size, decode_hidden)

    def make_xxx(self, feat_size, edge_hidden,
                 latent_sizes, neighborhood_sizes, neighbors,
                 timesteps, combine_hidden, target_size, pos_dim,
                 time_encoder):
        #TODO
        self.int = intr.TemporalInteraction(feat_size, edge_hidden,
                                            latent_sizes, neighborhood_sizes, neighbors,
                                            timesteps, combine_hidden, target_size, pos_dim,
                                            time_encoder)

    def run_tpc(self, data, ids, pos, ts, flat_pred_ts, space_dist_fn, time_dist_fn, target_dist_fn):
        #TODO
        out = self.int(data, ids, pos, ts, flat_pred_ts, space_dist_fn, time_dist_fn, target_dist_fn)
        return out

    def encode_targets(self, data, ids, ts, mask, pred_ids, pred_ts):
        pf = parse_frame(data)
        pos = pf['pos']
        space_dist_fn = partial(dists.space, mask, ts)
        time_dist_fn = partial(dists.time, mask, ids, self.time_encoder.encode)
        # Target info
        expand_pred_ts = pred_ts.unsqueeze(-1)
        expand_pred_ids = pred_ids.unsqueeze(-2)
        expand_pred_ts, expand_pred_ids = torch.broadcast_tensors(expand_pred_ts, expand_pred_ids)
        flat_pred_ts = expand_pred_ts.reshape(expand_pred_ts.size(0), -1)
        flat_pred_ids = expand_pred_ids.reshape(expand_pred_ids.size(0), -1)
        target_dist_fn = partial(dists.target, mask, flat_pred_ids, ids, self.time_encoder.encode)
        out = self.run_tpc(data, ids, pos, ts, flat_pred_ts, space_dist_fn, time_dist_fn, target_dist_fn)
        target_feats = out.view(expand_pred_ts.size()+(out.size(-1),))
        return target_feats


class SC2Blank(SC2TPC):
    def make_tpc(self, feat_size, weight_hidden, c_mid, final_hidden,
                 latent_sizes, neighborhood_sizes, neighbors,
                 timesteps, combine_hidden, target_size, pos_dim,
                 time_encoder):
        self.tpc = tpc.TemporalBlank(feat_size, weight_hidden, c_mid, final_hidden,
                                     latent_sizes, neighborhood_sizes, neighbors,
                                     timesteps, combine_hidden, target_size, pos_dim,
                                     time_encoder)


class SC2Zero(SC2TPC):
    def make_tpc(self, feat_size, weight_hidden, c_mid, final_hidden,
                 latent_sizes, neighborhood_sizes, neighbors,
                 timesteps, combine_hidden, target_size, pos_dim,
                 time_encoder):
        self.tpc = tpc.TemporalZero(feat_size, weight_hidden, c_mid, final_hidden,
                                    latent_sizes, neighborhood_sizes, neighbors,
                                    timesteps, combine_hidden, target_size, pos_dim,
                                    time_encoder)


class SC2GC(_SC2Common):
    def __init__(self,
                 neighborhood_sizes,
                 latent_sizes,
                 target_size,
                 combine_hidden,
                 decode_hidden,
                 neighbors, timesteps):
        super().__init__()
        feat_size = 16
        pos_dim = 2
        self.time_encoder = encodings.PeriodEncoding(8, 10)
        self.tgc = tpc.TemporalGraphConv(feat_size, latent_sizes, neighborhood_sizes,
                                         combine_hidden, target_size, neighbors, timesteps,
                                         pos_dim, self.time_encoder, 'time')
        self.make_decoders(target_size, decode_hidden)

    def encode_targets(self, data, ids, ts, mask, pred_ids, pred_ts):
        pf = parse_frame(data)
        pos = pf['pos']
        space_dist_fn = partial(dists.space, mask, ts)
        time_dist_fn = partial(dists.time, mask, ids, self.time_encoder.encode)
        # Target info
        expand_pred_ts = pred_ts.unsqueeze(-1)
        expand_pred_ids = pred_ids.unsqueeze(-2)
        expand_pred_ts, expand_pred_ids = torch.broadcast_tensors(expand_pred_ts, expand_pred_ids)
        flat_pred_ts = expand_pred_ts.reshape(expand_pred_ts.size(0), -1)
        flat_pred_ids = expand_pred_ids.reshape(expand_pred_ids.size(0), -1)
        target_dist_fn = partial(dists.target, mask, flat_pred_ids, ids, self.time_encoder.encode)
        out = self.tgc(data, ids, pos, ts, ts, flat_pred_ts, space_dist_fn, time_dist_fn, target_dist_fn)
        target_feats = out.view(expand_pred_ts.size()+(out.size(-1),))
        return target_feats


"""
class SC2Mink(_SC2Common):
    def __init__(self,
                 neighborhood_sizes = [2**4, 2**5, 2**5],
                 latent_sizes = [2**5, 2**6, 2**6],
                 target_size = 2**6,
                 combine_hidden = [2**6, 2**6],
                 weight_hidden = [2**5, 2**5],
                 c_mid = 2**5,
                 final_hidden = [2**6, 2**6],
                 decode_hidden = [2**6, 2**6, 2**6],
                 kernel_size = [21, 21, 17],
                 neighbors=8, timesteps=8):
        super().__init__()
        feat_size = 16
        self.qsize = 0.1
        self.kernel_size = kernel_size
        self.make_encoders(feat_size, neighborhood_sizes, latent_sizes, target_size,
                           neighbors, combine_hidden, timesteps, weight_hidden,
                           c_mid, final_hidden)

        self.make_decoders(target_size, decode_hidden)

    def get_args(self, item):
        keys = ('quant_data', 'quant_pts', 'quant_ids', 'quant_mask', 'pred_ids', 'pred_ts')
        return [item[k] for k in keys]

    def make_encoders(self, feat_size, neighborhood_sizes, latent_sizes, target_size,
                      neighbors, combine_hidden, timesteps, weight_hidden,
                      c_mid, final_hidden):
        self.minks = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.nonlins = nn.ModuleList()
        axis_types = [ME.RegionType.HYPER_CUBE, ME.RegionType.HYPER_CUBE, ME.RegionType.HYPER_CROSS]
        kernel = ME.KernelGenerator(self.kernel_size, region_type=ME.RegionType.HYPER_CUBE,
                                    axis_types=None, dimension=3)
        in_size = feat_size
        assert len(latent_sizes) == len(neighborhood_sizes)
        for ls, n_sz in zip(latent_sizes, neighborhood_sizes):
            mink = ME.MinkowskiConvolution(
                in_channels=in_size, out_channels=ls, dimension=3,
                kernel_generator=kernel)
            self.minks.append(mink)
            self.norms.append(ME.MinkowskiBatchNorm(ls))
            self.nonlins.append(ME.MinkowskiReLU())
            in_size = ls
        self.target_mink = ME.MinkowskiConvolution(
            in_channels=ls, out_channels=target_size, kernel_size=self.kernel_size[2],
            dimension=1)

    def forward(self, quant_data, quant_pts, quant_ids, quant_mask, pred_ids, pred_ts):
        enc_data = self.encode(quant_data, quant_pts)
        target_feats = self.calc_targets(enc_data, quant_pts, quant_ids, pred_ids, pred_ts)
        preds = self.predict(target_feats)
        return self.assemble_frame(*preds)

    def encode(self, data, pts):
        sparse_in = ME.SparseTensor(features=data, coordinates=pts)
        for mink, norm, nonlin in zip(self.minks, self.norms, self.nonlins):
            sparse_out = norm(nonlin(mink(sparse_in)))
            sparse_in = sparse_out
        return sparse_out.features

    def calc_targets(self, data, pts, sel_ids, in_pred_ids, pred_ts):
        ts = pts[:, [3]]
        pred_ids = in_pred_ids.unsqueeze(-1)
        pred_batch = torch.arange(pred_ids.size(0), device=pred_ids.device).view(-1, 1, 1)
        pred_batch, pred_ids = torch.broadcast_tensors(pred_batch, pred_ids)
        flat_pred_batch = pred_batch.reshape(-1, pred_batch.size(-1))
        flat_pred_ids = pred_ids.reshape(-1, pred_ids.size(-1))
        full_pred_ids = torch.cat((flat_pred_batch, flat_pred_ids), dim=-1)
        full_pred_ts = pred_ts[flat_pred_batch.squeeze(-1)]
        all_feats = [] ; all_pts = [] ; all_pred_ts = []
        batch_id_list = []
        for pred_id in full_pred_ids:
            mask = (pred_id==sel_ids).all(dim=-1)
            if mask.sum() > 0:
                all_feats.append(data[mask])
                _t = ts[mask]
                _bid = pred_id[0].item()
                _iid = (in_pred_ids[_bid]==pred_id[1]).nonzero(as_tuple=False)[0,0].item()
                batch_id_list.append((_bid, _iid))
                all_pts.append(ts[mask])
                all_pred_ts.append(pred_ts[_bid])
        batch_pred_ts = []
        for unit_idx, pred_t in enumerate(all_pred_ts):
            pred_t = pred_t.unsqueeze(-1)
            unit_idx = torch.tensor([unit_idx], device=pred_t.device).unsqueeze(-1)
            comb_t = torch.cat(torch.broadcast_tensors(unit_idx, pred_t), dim=-1)
            batch_pred_ts.append(comb_t)
        batch_pred_ts = torch.cat(batch_pred_ts, dim=0)
        hist_pts, hist_feats = ME.utils.sparse_collate(all_pts, all_feats)
        tensor_in = ME.SparseTensor(features=hist_feats, coordinates=hist_pts.to(hist_feats.device))
        _coords = batch_pred_ts.int()
        out = self.target_mink(tensor_in, coordinates=_coords)
        tgt_size = (in_pred_ids.size(0), in_pred_ids.size(1), pred_ts.size(1), out.features.size(-1))
        target_feats = torch.zeros(tgt_size, device=out.device)
        batch_id_map = dict(enumerate(batch_id_list))
        for pred_coord, pred_row in zip(out.coordinates, out.features):
            out_coords = batch_id_map[pred_coord[0].item()]
            batch = out_coords[0]
            t_idx = (pred_ts[batch]==pred_coord[1]).nonzero(as_tuple=False)[0,0].item()
            final_idx = out_coords+(t_idx,)
            target_feats[final_idx] = pred_row
        target_feats = target_feats.permute(0, 2, 1, 3).contiguous()
        return target_feats
"""


class SC2TPCSingle(_SC2Common):
    def __init__(self,
                 neighborhood_sizes,
                 latent_sizes,
                 target_size,
                 combine_hidden,
                 weight_hidden,
                 c_mid,
                 final_hidden,
                 decode_hidden,
                 neighbors, timesteps, time_factor):
        super().__init__()
        feat_size = 16
        self.time_factor = time_factor
        self.make_encoders(feat_size, neighborhood_sizes, latent_sizes, target_size,
                           neighbors, combine_hidden, timesteps, weight_hidden,
                           c_mid, final_hidden)

        self.make_decoders(target_size, decode_hidden)

    def make_encoders(self, feat_size, neighborhood_sizes, latent_sizes, target_size,
                      neighbors, combine_hidden, timesteps, weight_hidden,
                      c_mid, final_hidden):
        self.time_encoder = encodings.PeriodEncoding(8, 10)
        self.dist_convs = nn.ModuleList()
        self.combine_mlps = nn.ModuleList()
        in_size = feat_size
        default_args = {'weight_hidden': weight_hidden, 'c_mid': c_mid,
                        'final_hidden': final_hidden}
        assert len(latent_sizes) == len(neighborhood_sizes)
        for ls, n_sz in zip(latent_sizes, neighborhood_sizes):
            # Space neighborhood
            args = default_args.copy()
            args.update({'neighbors': neighbors, 'c_in': in_size, 'c_out': n_sz,
                         'dim': 3})
            pc = pointconv.PointConv(**args)
            self.dist_convs.append(pc)
            # MLP to next layer
            mlp_args = {'in_size': in_size+n_sz, 'out_size': ls,
                        'hidden_sizes': combine_hidden, 'reduction': 'none'}
            pn = SetTransform(**mlp_args)
            self.combine_mlps.append(pn)
            in_size = ls
        # Target conv
        args = default_args.copy()
        args = {'weight_hidden': [x*2 for x in weight_hidden],
                'c_mid': c_mid*2,
                'final_hidden': [x*2 for x in final_hidden],
                'neighbors': timesteps, 'c_in': ls, 'c_out': target_size,
                'dim': self.time_encoder.out_dim}
        self.target_conv = pointconv.PointConv(**args)

    def encode(self, data, ids, ts, mask):
        pf = parse_frame(data)
        # Add time to unit positions
        pos_t = torch.cat((pf['pos'], ts.unsqueeze(-1)), dim=-1)
        # Initial setup to execute network
        # Combined convolution operates on XYZ+T of units
        points = pos_t
        spacetime_dist_fn = partial(dists.spacetime, mask, self.time_factor)
        # Time convolution
        time_points = ts
        time_dist_fn = partial(dists.time, mask, ids, self.time_encoder.encode)
        space_in = data
        for dist, comb in \
                zip(self.dist_convs, self.combine_mlps):
            # Calculate spatial convolution
            space_nei = dist(points, points, space_in, spacetime_dist_fn)
            combined = torch.cat([space_in, space_nei], dim=2)
            # Construct input to next space conv by appending time conv
            # output
            space_in = comb(combined)
        return space_in

    def calc_targets(self, data, ids, ts, mask, pred_ids, pred_ts):
        expand_pred_ts = pred_ts.unsqueeze(-1)
        expand_pred_ids = pred_ids.unsqueeze(-2)
        expand_pred_ts, expand_pred_ids = torch.broadcast_tensors(expand_pred_ts, expand_pred_ids)
        flat_pred_ts = expand_pred_ts.reshape(expand_pred_ts.size(0), -1)
        flat_pred_ids = expand_pred_ids.reshape(expand_pred_ids.size(0), -1)
        target_dist_fn = partial(dists.target, mask, flat_pred_ids, ids, self.time_encoder.encode)
        target_feats = self.target_conv(flat_pred_ts, ts, data, target_dist_fn)
        # Restore dimensions
        target_feats = target_feats.view(expand_pred_ts.size()+(target_feats.size(-1),))
        return target_feats
