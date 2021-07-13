from . import pointconv
from . import graphconv
from . import encodings
from .pointnet import SetTransform

import torch
import torch.nn as nn

class TemporalPointConv(nn.Module):
    def __init__(self,
                 feat_size,
                 weight_hidden,
                 c_mid,
                 final_hidden,
                 latent_sizes,
                 neighborhood_sizes,
                 neighbors,
                 timesteps,
                 combine_hidden,
                 target_size,
                 pos_dim,
                 time_encoder
                 ):
        super().__init__()
        self.pos_dim = pos_dim
        self.time_encoder = time_encoder
        self._make_modules(feat_size, weight_hidden, c_mid, final_hidden,
                           latent_sizes, neighborhood_sizes,
                           neighbors, timesteps, combine_hidden, target_size)

    def _make_modules(self, feat_size, weight_hidden, c_mid, final_hidden,
                      latent_sizes, neighborhood_sizes, neighbors, timesteps,
                      combine_hidden, target_size):
        self.space_convs = nn.ModuleList()
        self.time_convs = nn.ModuleList()
        self.combine_mlps = nn.ModuleList()
        in_size = feat_size
        neighbor_attn = 0
        default_args = {'weight_hidden': weight_hidden, 'c_mid': c_mid,
                        'final_hidden': final_hidden, 'attn_heads': neighbor_attn}
        assert len(latent_sizes) == len(neighborhood_sizes)
        for ls, n_sz in zip(latent_sizes, neighborhood_sizes):
            # Space neighborhood
            args = default_args.copy()
            args.update({'neighbors': neighbors, 'c_in': in_size, 'c_out': n_sz,
                         'dim': self.pos_dim})
            pc = pointconv.PointConv(**args)
            self.space_convs.append(pc)
            # Time neighborhood
            args = default_args.copy()
            args.update({'neighbors': timesteps, 'c_in': in_size+n_sz,
                         'c_out': n_sz, 'dim': self.time_encoder.out_dim})
            pc = pointconv.PointConv(**args)
            self.time_convs.append(pc)
            # MLP to next layer
            mlp_args = {'in_size': in_size+2*n_sz, 'out_size': ls,
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

    def forward(self, data, ids, space_pts, time_pts, query_pts,
                space_dist_fn=None, time_dist_fn=None, target_dist_fn=None,
                space_dist_data=None, time_dist_data=None, target_dist_data=None):
        out_data = self.encode_input(data, ids, space_pts, time_pts, space_dist_fn, time_dist_fn, space_dist_data, time_dist_data)
        query_feats = self.encode_queries(out_data, time_pts, query_pts, target_dist_fn, target_dist_data)
        return query_feats

    def encode_input(self, data, ids, space_points, time_points, space_dist_fn, time_dist_fn, space_dist_data, time_dist_data):
        space_in = data
        for space, time, comb in \
                zip(self.space_convs, self.time_convs, self.combine_mlps):
            # Calculate spatial convolution
            space_nei = space(space_points, space_points, space_in, space_dist_fn, space_dist_data)
            # Combine input+output feats of space conv as input for time conv
            time_in = torch.cat([space_in, space_nei], dim=2)
            # Run time convolution
            time_nei = time(time_points, time_points, time_in, time_dist_fn, time_dist_data)
            combined = torch.cat([space_in, space_nei, time_nei], dim=2)
            # Construct input to next space conv by appending time conv
            # output
            space_in = comb(combined)
        return space_in

    def encode_queries(self, data, ts, query_ts, target_dist_fn, target_dist_data):
        target_feats = self.target_conv(query_ts, ts, data, target_dist_fn, target_dist_data)
        return target_feats


class TemporalGraphConv(nn.Module):
    def __init__(self,
                 feat_size,
                 latent_sizes,
                 neighborhood_sizes,
                 combine_hidden,
                 target_size,
                 neighbors,
                 timesteps,
                 pos_dim,
                 time_encoder
                 ):
        super().__init__()
        self.pos_dim = pos_dim
        self.time_encoder = time_encoder
        self._make_modules(feat_size, latent_sizes, neighborhood_sizes,
                           combine_hidden, target_size, neighbors, timesteps)

    def _make_modules(self, feat_size, latent_sizes, neighborhood_sizes, combine_hidden, target_size, neighbors, timesteps):
        self.space_convs = nn.ModuleList()
        self.time_convs = nn.ModuleList()
        self.combine_mlps = nn.ModuleList()
        in_size = feat_size
        neighbor_attn = 0
        default_args = {}
        assert len(latent_sizes) == len(neighborhood_sizes)
        for ls, n_sz in zip(latent_sizes, neighborhood_sizes):
            # Space neighborhood
            args = default_args.copy()
            args.update({'in_size': in_size, 'out_size': n_sz, 'rel_size': self.pos_dim,
                         'neighbor_count': neighbors})
            pc = graphconv.GraphConv(**args)
            self.space_convs.append(pc)
            # Time neighborhood
            args = default_args.copy()
            args.update({'in_size': in_size+n_sz, 'out_size': n_sz, 'rel_size': self.time_encoder.out_dim,
                         'neighbor_count': timesteps})
            pc = graphconv.GraphConv(**args)
            self.time_convs.append(pc)
            # MLP to next layer
            mlp_args = {'in_size': in_size+2*n_sz, 'out_size': ls,
                        'hidden_sizes': combine_hidden, 'reduction': 'none'}
            pn = SetTransform(**mlp_args)
            self.combine_mlps.append(pn)
            in_size = ls
        # Target conv
        args = {'in_size': ls, 'out_size': target_size, 'rel_size': self.time_encoder.out_dim,
                'neighbor_count': timesteps}
        self.target_conv = graphconv.GraphConv(**args)

    def forward(self, data, ids, space_pts, time_pts, query_pts,
                space_dist_fn=None, time_dist_fn=None, target_dist_fn=None,
                space_dist_data=None, time_dist_data=None, target_dist_data=None):
        out_data = self.encode_input(data, ids, space_pts, time_pts, space_dist_fn, time_dist_fn, space_dist_data, time_dist_data)
        query_feats = self.encode_queries(out_data, time_pts, query_pts, target_dist_fn, target_dist_data)
        return query_feats

    def encode_input(self, data, ids, space_points, time_points, space_dist_fn, time_dist_fn, space_dist_data, time_dist_data):
        space_in = data
        for space, time, comb in \
                zip(self.space_convs, self.time_convs, self.combine_mlps):
            # Calculate spatial convolution
            space_nei = space(space_points, space_points, space_in, space_dist_fn, space_dist_data)
            # Combine input+output feats of space conv as input for time conv
            time_in = torch.cat([space_in, space_nei], dim=2)
            # Run time convolution
            time_nei = time(time_points, time_points, time_in, time_dist_fn, time_dist_data)
            combined = torch.cat([space_in, space_nei, time_nei], dim=2)
            # Construct input to next space conv by appending time conv
            # output
            space_in = comb(combined)
        return space_in

    def encode_queries(self, data, ts, query_ts, target_dist_fn, target_dist_data):
        target_feats = self.target_conv(query_ts, ts, data, target_dist_fn, target_dist_data)
        return target_feats
