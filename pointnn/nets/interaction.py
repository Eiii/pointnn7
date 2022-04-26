import torch
import torch.nn as nn

from .pointnet import SetTransform
from .pointconv import calc_neighbor_info


# TODO: unused params?
class TemporalInteraction(nn.Module):
    def __init__(self,
                 feat_size,
                 edge_hidden,
                 latent_sizes,
                 neighborhood_sizes,
                 neighbors,
                 timesteps,
                 combine_hidden,
                 target_size,
                 pos_dim,
                 time_encoder,
                 query_type='time'
                 ):
        super().__init__()
        self.pos_dim = pos_dim
        self.time_encoder = time_encoder
        self.query_type = query_type
        self._make_modules(feat_size, edge_hidden,
                           latent_sizes, neighborhood_sizes,
                           neighbors, timesteps, combine_hidden, target_size)

    def _make_modules(self, feat_size, edge_hidden,
                      latent_sizes, neighborhood_sizes, neighbors, timesteps,
                      combine_hidden, target_size):
        self.space_convs = nn.ModuleList()
        self.time_convs = nn.ModuleList()
        self.combine_mlps = nn.ModuleList()
        in_size = feat_size
        neighbor_attn = 0
        default_args = {'edge_hidden': edge_hidden}
        assert len(latent_sizes) == len(neighborhood_sizes)
        for ls, n_sz in zip(latent_sizes, neighborhood_sizes):
            # Space neighborhood
            args = default_args.copy()
            args.update({'neighbors': neighbors, 'c_in': in_size+self.pos_dim, 'c_out': n_sz,
                         'dim': self.pos_dim})
            pc = InteractionNetworkNeighborhood(**args)
            self.space_convs.append(pc)
            # Time neighborhood
            args = default_args.copy()
            args.update({'neighbors': timesteps, 'c_in': in_size+n_sz+self.time_encoder.out_dim,
                         'c_out': n_sz, 'dim': self.time_encoder.out_dim})
            pc = InteractionNetworkNeighborhood(**args)
            self.time_convs.append(pc)
            # MLP to next layer
            mlp_args = {'in_size': in_size+2*n_sz, 'out_size': ls,
                        'hidden_sizes': combine_hidden, 'reduction': 'none'}
            pn = SetTransform(**mlp_args)
            self.combine_mlps.append(pn)
            in_size = ls
        # Target conv
        if self.query_type == 'time':
            query_dim = self.time_encoder.out_dim
            query_encoder = self.time_encoder
        elif self.query_type == 'space':
            query_dim = self.pos_dim
            query_encoder = None
        args = default_args.copy()
        args = {'edge_hidden': [x*2 for x in edge_hidden],
                'neighbors': timesteps, 'c_in': ls+query_dim, 'c_out': target_size,
                'dim': query_dim,
                'key_feats': 'query',
                'pos_encoder': query_encoder}
        self.target_conv = InteractionNetworkNeighborhood(**args)

    def forward(self, data, ids, space_pts, time_pts, query_pts,
                space_dist_fn=None, time_dist_fn=None, target_dist_fn=None,
                space_dist_data=None, time_dist_data=None, target_dist_data=None):
        out_data = self.encode_input(data, ids, space_pts, time_pts, space_dist_fn, time_dist_fn, space_dist_data, time_dist_data)
        if self.query_type == 'time':
            in_query_pts = time_pts
        elif self.query_type == 'space':
            in_query_pts = space_pts
        query_feats = self.encode_queries(out_data, in_query_pts, query_pts, target_dist_fn, target_dist_data)
        return query_feats

    def encode_input(self, data, ids, space_points, time_points, space_dist_fn, time_dist_fn, space_dist_data, time_dist_data):
        key_feats = data
        for space, time, comb in \
                zip(self.space_convs, self.time_convs, self.combine_mlps):
            # Calculate spatial convolution
            space_in = torch.cat([space_points, key_feats], dim=-1)
            space_nei = space(space_points, space_points, space_in, space_dist_fn, space_dist_data)
            # Combine input+output feats of space conv as input for time conv
            enc_time_points = self.time_encoder.encode(time_points.view([*time_points.shape, 1, 1])).squeeze(-2)
            time_in = torch.cat([enc_time_points, key_feats, space_nei], dim=2)
            # Run time convolution
            time_nei = time(time_points, time_points, time_in, time_dist_fn, time_dist_data)
            combined = torch.cat([key_feats, space_nei, time_nei], dim=2)
            # Construct input to next space conv by appending time conv
            # output
            key_feats = comb(combined)
        return key_feats

    def encode_queries(self, data, ts, query_ts, target_dist_fn, target_dist_data):
        if self.query_type == 'time': #HACK
            enc_ts = self.time_encoder.encode(ts.view([*ts.shape, 1, 1])).squeeze(-2)
        elif self.query_type == 'space':
            enc_ts = ts
        target_in = torch.cat([enc_ts, data], dim=-1)
        target_feats = self.target_conv(query_ts, ts, target_in, target_dist_fn, target_dist_data)
        return target_feats

class TIntNoSpace(nn.Module):
    def __init__(self,
                 feat_size,
                 edge_hidden,
                 latent_sizes,
                 neighborhood_sizes,
                 neighbors,
                 timesteps,
                 combine_hidden,
                 target_size,
                 pos_dim,
                 time_encoder,
                 query_type='time'
                 ):
        super().__init__()
        self.pos_dim = pos_dim
        self.time_encoder = time_encoder
        self.query_type = query_type
        self._make_modules(feat_size, edge_hidden,
                           latent_sizes, neighborhood_sizes,
                           neighbors, timesteps, combine_hidden, target_size)

    def _make_modules(self, feat_size, edge_hidden,
                      latent_sizes, neighborhood_sizes, neighbors, timesteps,
                      combine_hidden, target_size):
        self.n_sizes = list()
        self.time_convs = nn.ModuleList()
        self.combine_mlps = nn.ModuleList()
        in_size = feat_size
        neighbor_attn = 0
        default_args = {'edge_hidden': edge_hidden}
        assert len(latent_sizes) == len(neighborhood_sizes)
        for ls, n_sz in zip(latent_sizes, neighborhood_sizes):
            # Space neighborhood
            self.n_sizes.append(n_sz)
            # Time neighborhood
            args = default_args.copy()
            args.update({'neighbors': timesteps, 'c_in': in_size+n_sz+self.time_encoder.out_dim,
                         'c_out': n_sz, 'dim': self.time_encoder.out_dim})
            pc = InteractionNetworkNeighborhood(**args)
            self.time_convs.append(pc)
            # MLP to next layer
            mlp_args = {'in_size': in_size+2*n_sz, 'out_size': ls,
                        'hidden_sizes': combine_hidden, 'reduction': 'none'}
            pn = SetTransform(**mlp_args)
            self.combine_mlps.append(pn)
            in_size = ls
        # Target conv
        if self.query_type == 'time':
            query_dim = self.time_encoder.out_dim
            query_encoder = self.time_encoder
        elif self.query_type == 'space':
            query_dim = self.pos_dim
            query_encoder = None
        args = default_args.copy()
        args = {'edge_hidden': [x*2 for x in edge_hidden],
                'neighbors': timesteps, 'c_in': ls+query_dim, 'c_out': target_size,
                'dim': query_dim,
                'key_feats': 'query',
                'pos_encoder': query_encoder}
        self.target_conv = InteractionNetworkNeighborhood(**args)

    def forward(self, data, ids, space_pts, time_pts, query_pts,
                space_dist_fn=None, time_dist_fn=None, target_dist_fn=None,
                space_dist_data=None, time_dist_data=None, target_dist_data=None):
        out_data = self.encode_input(data, ids, space_pts, time_pts, space_dist_fn, time_dist_fn, space_dist_data, time_dist_data)
        if self.query_type == 'time':
            in_query_pts = time_pts
        elif self.query_type == 'space':
            in_query_pts = space_pts
        query_feats = self.encode_queries(out_data, in_query_pts, query_pts, target_dist_fn, target_dist_data)
        return query_feats

    def encode_input(self, data, ids, space_points, time_points, space_dist_fn, time_dist_fn, space_dist_data, time_dist_data):
        key_feats = data
        for n_size, time, comb in \
                zip(self.n_sizes, self.time_convs, self.combine_mlps):
            # Calculate spatial convolution
            sz = [*key_feats.size()[:2], n_size]
            space_nei = torch.zeros(sz, device=key_feats.device)
            # Combine input+output feats of space conv as input for time conv
            enc_time_points = self.time_encoder.encode(time_points.view([*time_points.shape, 1, 1])).squeeze(-2)
            time_in = torch.cat([enc_time_points, key_feats, space_nei], dim=2)
            # Run time convolution
            time_nei = time(time_points, time_points, time_in, time_dist_fn, time_dist_data)
            combined = torch.cat([key_feats, space_nei, time_nei], dim=2)
            # Construct input to next space conv by appending time conv
            # output
            key_feats = comb(combined)
        return key_feats

    def encode_queries(self, data, ts, query_ts, target_dist_fn, target_dist_data):
        if self.query_type == 'time': #HACK
            enc_ts = self.time_encoder.encode(ts.view([*ts.shape, 1, 1])).squeeze(-2)
        elif self.query_type == 'space':
            enc_ts = ts
        target_in = torch.cat([enc_ts, data], dim=-1)
        target_feats = self.target_conv(query_ts, ts, target_in, target_dist_fn, target_dist_data)
        return target_feats


class InteractionNetworkNeighborhood(nn.Module):
    def __init__(self,
                 neighbors,
                 c_in,
                 edge_hidden,
                 c_out,
                 dim=3,
                 dist_fn=None,
                 key_feats='self',
                 pos_encoder=None
                 ):
        super().__init__()
        self.dist_fn = dist_fn
        self.neighbor_count = neighbors
        assert key_feats in ('self', 'query')
        self.key_feats = key_feats
        self.pos_encoder = pos_encoder
        if key_feats == 'self':
            in_sz = c_in*2
        elif key_feats == 'query':
            in_sz = c_in+dim
        self.edge_conv = SetTransform(in_size=in_sz, out_size=c_out,
                                      hidden_sizes=list(edge_hidden),
                                      reduction='none')

    def forward(self, keys, points, feats, dist_fn=None, dist_data=None):
        if dist_data is not None:
            n_idxs, neighbor_rel, neighbor_valid = dist_data
            bb = torch.arange(n_idxs.size(0), device=n_idxs.device).view(-1, 1, 1)
            neighbor_feats = feats[bb, n_idxs]
        else:
            neighbor_rel, neighbor_feats, neighbor_valid = \
                calc_neighbor_info(keys, points, feats, self.neighbor_count,
                                   dist_fn or self.dist_fn)
        edges = self.calc_edges(feats, keys, neighbor_feats)
        neighbor_valid = neighbor_valid.unsqueeze(-1)
        masked_edges = edges * neighbor_valid
        reduced_edges = masked_edges.sum(dim=2)
        return reduced_edges

    def calc_edges(self, feats, keys, neighbor_feats):
        if self.key_feats == 'self':
            exp_feats = feats.unsqueeze(2).expand(-1, -1, neighbor_feats.size(2), -1)
        elif self.key_feats == 'query':
            if self.pos_encoder is not None:
                enc_keys = self.pos_encoder.encode(keys.view([*keys.shape, 1, 1]))
            else:
                enc_keys = keys.unsqueeze(-2)
            exp_feats = enc_keys.expand(-1, -1, neighbor_feats.size(2), -1)
        pairs = torch.cat((exp_feats, neighbor_feats), dim=-1)
        edges = self.edge_conv(pairs)
        return edges
