from .. import pointnet
from .. import encodings
from .. import tpc
from .. import interaction as intr
from ..base import Network

from functools import partial

import torch


class _TrafficCommon(Network):
    def get_args(self, item):
        xs = ['hist_t', 'hist_id', 'hist_pos', 'hist_data', 'id_dist', 'id_adj', 'hist_mask',
              'tgt_t', 'tgt_id', 'tgt_mask']
        args = [item[x] for x in xs]
        return args

    def _calc_means(self, tgt_id, hist_id, hist_data):
        all_ids = tgt_id.unique()
        means = torch.zeros_like(tgt_id, dtype=torch.float)
        for id_ in all_ids:
            hist_mask = (hist_id == id_)
            id_data = hist_data * hist_mask.unsqueeze(-1)
            id_avg = id_data.sum(dim=-1).sum(dim=-1) / hist_mask.sum(dim=-1)
            tgt_idxs = (tgt_id == id_).nonzero(as_tuple=True)
            tgt_means = id_avg[tgt_idxs[0]]
            means[tgt_idxs] = tgt_means
        return means


class TrafficTPC(_TrafficCommon):
    tpc_class = tpc.TemporalPointConv
    def __init__(self,
                 neighborhood_sizes = [2**4, 2**5, 2**5],
                 latent_sizes = [2**5, 2**6, 2**6],
                 target_size = 2**6,
                 combine_hidden = [2**6, 2**6],
                 weight_hidden = [2**5, 2**5],
                 c_mid = 2**5,
                 final_hidden = [2**6, 2**6],
                 decode_hidden = [2**6, 2**6, 2**6],
                 neighbors=8, timesteps=12,
                 mean_delta=False,
                 heads=0):
        super().__init__()
        feat_size = 1
        pos_dim = 2+1
        self.mean_delta = mean_delta
        self.time_encoder = encodings.DirectEncoding()
        self.neighbors = neighbors
        self.timesteps = timesteps
        self.tpc = self.tpc_class(feat_size, weight_hidden, c_mid, final_hidden,
                                  latent_sizes, neighborhood_sizes, neighbors,
                                  timesteps, timesteps, combine_hidden,
                                  target_size, pos_dim, self.time_encoder, heads)
        self.make_decoders(target_size, decode_hidden)

    def make_decoders(self, in_size, decode_hidden):
        args = {'in_size': in_size, 'out_size': 1,
                'hidden_sizes': decode_hidden, 'reduction': 'none'}
        self.pred_net = pointnet.SetTransform(**args)

    def decode_queries(self, feats):
        pred = self.pred_net(feats)
        return pred

    def forward(self, hist_t, hist_id, hist_pos, hist_data, id_dist, id_adj, hist_mask,
                tgt_t, tgt_id, tgt_mask):
        space_dist_data = space_neighbors(self.neighbors, hist_t, hist_id, hist_pos, hist_mask, id_dist, id_adj)
        time_dist_data = time_neighbors(self.timesteps, 5, hist_t, hist_id, hist_mask)
        query_dist_fn = partial(query, self.time_encoder.encode, tgt_mask, hist_mask,
                                tgt_id, hist_id)
        hist_data = hist_data.unsqueeze(-1)
        query_feats = self.tpc(hist_data, hist_id, hist_pos, hist_t, tgt_t,
                               space_dist_data=space_dist_data, time_dist_data=time_dist_data, target_dist_fn=query_dist_fn)
        pred = self.decode_queries(query_feats)
        if self.mean_delta:
            means = self._calc_means(tgt_id, hist_id, hist_data)
            pred = pred+means.unsqueeze(-1)
        return pred


class TrafficTPCNoSpace(TrafficTPC):
    tpc_class = tpc.TPCNoSpace


class TrafficInteraction(_TrafficCommon):
    int_class = intr.TemporalInteraction
    def __init__(self,
                 neighborhood_sizes = [2**4, 2**5, 2**5],
                 latent_sizes = [2**5, 2**6, 2**6],
                 target_size = 2**6,
                 combine_hidden = [2**6, 2**6],
                 edge_hidden = [2**5, 2**5],
                 decode_hidden = [2**6, 2**6, 2**6],
                 neighbors=8, timesteps=12,
                 mean_delta=False):
        super().__init__()
        feat_size = 1
        pos_dim = 2
        self.mean_delta = mean_delta
        self.time_encoder = encodings.DirectEncoding()
        self.neighbors = neighbors
        self.timesteps = timesteps
        self.int = self.int_class(feat_size, edge_hidden,
                                  latent_sizes, neighborhood_sizes, neighbors,
                                  timesteps, combine_hidden, target_size, pos_dim,
                                  self.time_encoder)
        self.make_decoders(target_size, decode_hidden)

    def make_decoders(self, in_size, decode_hidden):
        args = {'in_size': in_size, 'out_size': 1,
                'hidden_sizes': decode_hidden, 'reduction': 'none'}
        self.pred_net = pointnet.SetTransform(**args)

    def decode_queries(self, feats):
        pred = self.pred_net(feats)
        return pred

    def forward(self, hist_t, hist_id, hist_pos, hist_data, id_dist, id_adj, hist_mask,
                tgt_t, tgt_id, tgt_mask):
        space_dist_data = space_neighbors(self.neighbors, hist_t, hist_id, hist_pos, hist_mask, id_dist, id_adj)
        time_dist_data = time_neighbors(self.timesteps, 5, hist_t, hist_id, hist_mask)
        query_dist_fn = partial(query, self.time_encoder.encode, tgt_mask, hist_mask,
                                tgt_id, hist_id)
        hist_data = hist_data.unsqueeze(-1)
        query_feats = self.int(hist_data, hist_id, hist_pos, hist_t, tgt_t,
                               space_dist_data=space_dist_data, time_dist_data=time_dist_data, target_dist_fn=query_dist_fn)
        pred = self.decode_queries(query_feats)
        if self.mean_delta:
            means = self._calc_means(tgt_id, hist_id, hist_data)
            pred = pred+means.unsqueeze(-1)
        return pred


class TrafficIntNoSpace(TrafficInteraction):
    int_class = intr.TIntNoSpace


class TrafficGraphConv(Network):
    def __init__(self,
                 neighborhood_sizes = [2**4, 2**5, 2**5],
                 latent_sizes = [2**5, 2**6, 2**6],
                 target_size = 2**6,
                 combine_hidden = [2**6, 2**6],
                 decode_hidden = [2**6, 2**6, 2**6],
                 neighbors=8, timesteps=12):
        super().__init__()
        feat_size = 1
        pos_dim = 2+1
        self.time_encoder = encodings.DirectEncoding()
        self.neighbors = neighbors
        self.timesteps = timesteps
        self.tgc = tpc.TemporalGraphConv(feat_size, latent_sizes, neighborhood_sizes,
                                         combine_hidden, target_size, neighbors, timesteps,
                                         pos_dim, self.time_encoder, 'time')
        self.make_decoders(target_size, decode_hidden)

    def get_args(self, item):
        xs = ['hist_t', 'hist_id', 'hist_pos', 'hist_data', 'id_dist', 'id_adj', 'hist_mask',
              'tgt_t', 'tgt_id', 'tgt_mask']
        return [item[x] for x in xs]

    def make_decoders(self, in_size, decode_hidden):
        args = {'in_size': in_size, 'out_size': 1,
                'hidden_sizes': decode_hidden, 'reduction': 'none'}
        self.pred_net = pointnet.SetTransform(**args)

    def decode_queries(self, feats):
        pred = self.pred_net(feats)
        return pred

    def forward(self, hist_t, hist_id, hist_pos, hist_data, id_dist, id_adj, hist_mask,
                tgt_t, tgt_id, tgt_mask):
        space_dist_data = space_neighbors(self.neighbors, hist_t, hist_id, hist_pos, hist_mask, id_dist, id_adj)
        time_dist_data = time_neighbors(self.timesteps, 5, hist_t, hist_id, hist_mask)
        query_dist_fn = partial(query, self.time_encoder.encode, tgt_mask, hist_mask,
                                tgt_id, hist_id)
        hist_data = hist_data.unsqueeze(-1)
        query_feats = self.tgc(hist_data, hist_id, hist_pos, hist_t, hist_t, tgt_t,
                               space_dist_data=space_dist_data,
                               time_dist_data=time_dist_data,
                               target_dist_fn=query_dist_fn)
        pred = self.decode_queries(query_feats)
        return pred

class TrafficSpectral(Network):
    def __init__(self,
                 neighborhood_sizes = [2**4, 2**5, 2**5],
                 latent_sizes = [2**5, 2**6, 2**6],
                 target_size = 2**6,
                 combine_hidden = [2**6, 2**6],
                 weight_hidden = [2**5, 2**5],
                 c_mid = 2**5,
                 final_hidden = [2**6, 2**6],
                 decode_hidden = [2**6, 2**6, 2**6],
                 neighbors=8, timesteps=12,
                 mean_delta=False,
                 eig_dims=20):
        super().__init__()
        feat_size = 1
        pos_dim = 2+1
        self.mean_delta = mean_delta
        self.time_encoder = encodings.DirectEncoding()
        self.neighbors = neighbors
        self.timesteps = timesteps
        self.tpc = tpc.TemporalSpectral(feat_size, weight_hidden, c_mid, final_hidden,
                                        latent_sizes, neighborhood_sizes, neighbors,
                                        timesteps, combine_hidden, target_size, pos_dim,
                                        self.time_encoder, eig_dims)
        self.make_decoders(target_size, decode_hidden)

    def get_args(self, item):
        xs = ['hist_t', 'hist_id', 'hist_pos', 'hist_data', 'id_dist', 'id_adj', 'hist_mask',
              'tgt_t', 'tgt_id', 'tgt_mask']
        args = [item[x] for x in xs]
        args.append(item['eig'])
        return args

    def make_decoders(self, in_size, decode_hidden):
        args = {'in_size': in_size, 'out_size': 1,
                'hidden_sizes': decode_hidden, 'reduction': 'none'}
        self.pred_net = pointnet.SetTransform(**args)

    def decode_queries(self, feats):
        pred = self.pred_net(feats)
        return pred

    def forward(self, hist_t, hist_id, hist_pos, hist_data, id_dist, id_adj, hist_mask,
                tgt_t, tgt_id, tgt_mask, eig):
        space_dist_data = space_neighbors(self.neighbors, hist_t, hist_id, hist_pos, hist_mask, id_dist, id_adj)
        time_dist_data = time_neighbors(self.timesteps, 5, hist_t, hist_id, hist_mask)
        query_dist_fn = partial(query, self.time_encoder.encode, tgt_mask, hist_mask,
                                tgt_id, hist_id)
        hist_data = hist_data.unsqueeze(-1)
        query_feats = self.tpc(hist_data, hist_id, hist_pos, hist_t, tgt_t, eig,
                               space_dist_data=space_dist_data, time_dist_data=time_dist_data, target_dist_fn=query_dist_fn)
        pred = self.decode_queries(query_feats)
        if self.mean_delta:
            means = self._calc_means(tgt_id, hist_id, hist_data)
            pred = pred+means.unsqueeze(-1)
        return pred

    def _calc_means(self, tgt_id, hist_id, hist_data):
        all_ids = tgt_id.unique()
        means = torch.zeros_like(tgt_id, dtype=torch.float)
        for id_ in all_ids:
            hist_mask = (hist_id == id_)
            id_data = hist_data * hist_mask.unsqueeze(-1)
            id_avg = id_data.sum(dim=-1).sum(dim=-1) / hist_mask.sum(dim=-1)
            tgt_idxs = (tgt_id == id_).nonzero(as_tuple=True)
            tgt_means = id_avg[tgt_idxs[0]]
            means[tgt_idxs] = tgt_means
        return means

class TrafficBlank(Network):
    def __init__(self,
                 neighborhood_sizes = [2**4, 2**5, 2**5],
                 latent_sizes = [2**5, 2**6, 2**6],
                 target_size = 2**6,
                 combine_hidden = [2**6, 2**6],
                 weight_hidden = [2**5, 2**5],
                 c_mid = 2**5,
                 final_hidden = [2**6, 2**6],
                 decode_hidden = [2**6, 2**6, 2**6],
                 neighbors=8, timesteps=12,
                 mean_delta=False,
                 eig_dims=20):
        super().__init__()
        feat_size = 1
        pos_dim = 2+1
        self.mean_delta = mean_delta
        self.time_encoder = encodings.DirectEncoding()
        self.neighbors = neighbors
        self.timesteps = timesteps
        self.tpc = tpc.TemporalBlank(feat_size, weight_hidden, c_mid, final_hidden,
                                        latent_sizes, neighborhood_sizes, neighbors,
                                        timesteps, combine_hidden, target_size, pos_dim,
                                        self.time_encoder)
        self.make_decoders(target_size, decode_hidden)

    def get_args(self, item):
        xs = ['hist_t', 'hist_id', 'hist_pos', 'hist_data', 'id_dist', 'id_adj', 'hist_mask',
              'tgt_t', 'tgt_id', 'tgt_mask']
        args = [item[x] for x in xs]
        args.append(item['eig'])
        return args

    def make_decoders(self, in_size, decode_hidden):
        args = {'in_size': in_size, 'out_size': 1,
                'hidden_sizes': decode_hidden, 'reduction': 'none'}
        self.pred_net = pointnet.SetTransform(**args)

    def decode_queries(self, feats):
        pred = self.pred_net(feats)
        return pred

    def forward(self, hist_t, hist_id, hist_pos, hist_data, id_dist, id_adj, hist_mask,
                tgt_t, tgt_id, tgt_mask, eig):
        space_dist_data = space_neighbors(self.neighbors, hist_t, hist_id, hist_pos, hist_mask, id_dist, id_adj)
        time_dist_data = time_neighbors(self.timesteps, 5, hist_t, hist_id, hist_mask)
        query_dist_fn = partial(query, self.time_encoder.encode, tgt_mask, hist_mask,
                                tgt_id, hist_id)
        hist_data = hist_data.unsqueeze(-1)
        query_feats = self.tpc(hist_data, hist_id, hist_pos, hist_t, tgt_t,
                               space_dist_data=space_dist_data, time_dist_data=time_dist_data, target_dist_fn=query_dist_fn)
        pred = self.decode_queries(query_feats)
        if self.mean_delta:
            means = self._calc_means(tgt_id, hist_id, hist_data)
            pred = pred+means.unsqueeze(-1)
        return pred

    def _calc_means(self, tgt_id, hist_id, hist_data):
        all_ids = tgt_id.unique()
        means = torch.zeros_like(tgt_id, dtype=torch.float)
        for id_ in all_ids:
            hist_mask = (hist_id == id_)
            id_data = hist_data * hist_mask.unsqueeze(-1)
            id_avg = id_data.sum(dim=-1).sum(dim=-1) / hist_mask.sum(dim=-1)
            tgt_idxs = (tgt_id == id_).nonzero(as_tuple=True)
            tgt_means = id_avg[tgt_idxs[0]]
            means[tgt_idxs] = tgt_means
        return means

class TrafficZero(Network):
    def __init__(self,
                 neighborhood_sizes = [2**4, 2**5, 2**5],
                 latent_sizes = [2**5, 2**6, 2**6],
                 target_size = 2**6,
                 combine_hidden = [2**6, 2**6],
                 weight_hidden = [2**5, 2**5],
                 c_mid = 2**5,
                 final_hidden = [2**6, 2**6],
                 decode_hidden = [2**6, 2**6, 2**6],
                 neighbors=8, timesteps=12,
                 mean_delta=False,
                 eig_dims=20):
        super().__init__()
        feat_size = 1
        pos_dim = 2+1
        self.mean_delta = mean_delta
        self.time_encoder = encodings.DirectEncoding()
        self.neighbors = neighbors
        self.timesteps = timesteps
        self.tpc = tpc.TemporalZero(feat_size, weight_hidden, c_mid, final_hidden,
                                        latent_sizes, neighborhood_sizes, neighbors,
                                        timesteps, combine_hidden, target_size, pos_dim,
                                        self.time_encoder)
        self.make_decoders(target_size, decode_hidden)

    def get_args(self, item):
        xs = ['hist_t', 'hist_id', 'hist_pos', 'hist_data', 'id_dist', 'id_adj', 'hist_mask',
              'tgt_t', 'tgt_id', 'tgt_mask']
        args = [item[x] for x in xs]
        args.append(item['eig'])
        return args

    def make_decoders(self, in_size, decode_hidden):
        args = {'in_size': in_size, 'out_size': 1,
                'hidden_sizes': decode_hidden, 'reduction': 'none'}
        self.pred_net = pointnet.SetTransform(**args)

    def decode_queries(self, feats):
        pred = self.pred_net(feats)
        return pred

    def forward(self, hist_t, hist_id, hist_pos, hist_data, id_dist, id_adj, hist_mask,
                tgt_t, tgt_id, tgt_mask, eig):
        space_dist_data = space_neighbors(self.neighbors, hist_t, hist_id, hist_pos, hist_mask, id_dist, id_adj)
        time_dist_data = time_neighbors(self.timesteps, 5, hist_t, hist_id, hist_mask)
        query_dist_fn = partial(query, self.time_encoder.encode, tgt_mask, hist_mask,
                                tgt_id, hist_id)
        hist_data = hist_data.unsqueeze(-1)
        query_feats = self.tpc(hist_data, hist_id, hist_pos, hist_t, tgt_t,
                               space_dist_data=space_dist_data, time_dist_data=time_dist_data, target_dist_fn=query_dist_fn)
        pred = self.decode_queries(query_feats)
        if self.mean_delta:
            means = self._calc_means(tgt_id, hist_id, hist_data)
            pred = pred+means.unsqueeze(-1)
        return pred

    def _calc_means(self, tgt_id, hist_id, hist_data):
        all_ids = tgt_id.unique()
        means = torch.zeros_like(tgt_id, dtype=torch.float)
        for id_ in all_ids:
            hist_mask = (hist_id == id_)
            id_data = hist_data * hist_mask.unsqueeze(-1)
            id_avg = id_data.sum(dim=-1).sum(dim=-1) / hist_mask.sum(dim=-1)
            tgt_idxs = (tgt_id == id_).nonzero(as_tuple=True)
            tgt_means = id_avg[tgt_idxs[0]]
            means[tgt_idxs] = tgt_means
        return means

def space_neighbors(k, hist_t, hist_id, hist_pos, hist_mask, id_dist, id_adj):
    all_ts = hist_t.unique()
    batches, samples = hist_t.shape
    device = hist_t.device
    final_valid = torch.zeros((batches, samples, k), device=device, dtype=torch.bool)
    vec_size = hist_pos.shape[-1] + 1
    final_vec = torch.zeros((batches, samples, k, vec_size), device=device, dtype=torch.float)
    final_idx = torch.zeros((batches, samples, k), device=device, dtype=torch.long)
    for this_t in all_ts:
        # Get indices for all samples from this timestep
        outer_idxs = (hist_t == this_t).nonzero(as_tuple=True)
        bid = outer_idxs[0] # Batch of each sample (flattened away)
        # Which samples are from the same batch? No cross-batch interactions
        # allowed
        cross_bids = torch.broadcast_tensors(*[bid.unsqueeze(x) for x in (0, 1)])
        valid_bid = (cross_bids[0] == cross_bids[1])
        # Use indices to get ID, position of each sample
        lcl_ids = hist_id[outer_idxs]
        lcl_pos = hist_pos[outer_idxs]
        lcl_mask = hist_mask[outer_idxs]
        # Create product tensor over sample IDs
        cross_ids = torch.broadcast_tensors(*[lcl_ids.unsqueeze(x) for x in (0, 1)])
        same_id = (cross_ids[0] == cross_ids[1]) # Samples can't be neighbors with themselves
        combined_idx = [bid, *cross_ids] # Combine batch + crossed IDs to create 'lookup' index tensor
        lcl_dist = id_dist[combined_idx] # Use index to lookup distance between each ID pair
        lcl_valid = id_adj[combined_idx] * valid_bid * ~same_id * lcl_mask.unsqueeze(0) * lcl_mask.unsqueeze(1) # Lookup adj. info + same batch? + different ID? to determine each pair's validity
        big_dist = lcl_dist.max()*1e2
        mod_lcl_dist = lcl_dist + (big_dist * ~lcl_valid) # Add big val to each dist so invalid ones get sorted to the end
        closest_dist, closest_idxs = mod_lcl_dist.topk(k, largest=False) # Get valid samples w/ smallest distance
        closest_pos = lcl_pos[closest_idxs] - lcl_pos.unsqueeze(1)
        closest_vec = torch.cat((closest_dist.unsqueeze(-1), closest_pos), dim=-1)
        final_valid[outer_idxs] = torch.gather(lcl_valid, 1, closest_idxs)
        final_idx[outer_idxs] = outer_idxs[1][closest_idxs]
        final_vec[outer_idxs] = closest_vec
    return final_idx, final_vec, final_valid

def time_neighbors(k, num_groups, hist_t, hist_id, hist_mask):
    all_ids = hist_id.unique()
    batches, samples = hist_id.shape
    device = hist_id.device
    final_valid = torch.zeros((batches, samples, k), device=device, dtype=torch.bool)
    final_vec = torch.zeros((batches, samples, k), device=device, dtype=torch.float)
    final_idx = torch.zeros((batches, samples, k), device=device, dtype=torch.long)
    group_size = len(all_ids)//num_groups
    for group_start in range(0, len(all_ids), group_size):
        id_group = all_ids[group_start:group_start+group_size]
        in_group = (hist_id.unsqueeze(-1) == id_group).any(-1)
        outer_idxs = in_group.nonzero(as_tuple=True)
        bid = outer_idxs[0] # Batch of each sample (flattened away)
        lcl_ids = hist_id[outer_idxs]
        lcl_ts = hist_t[outer_idxs]
        lcl_mask = hist_mask[outer_idxs]
        same_id = (lcl_ids.unsqueeze(0) == lcl_ids.unsqueeze(1))
        same_batch = (bid.unsqueeze(0) == bid.unsqueeze(1))
        same = same_id * same_batch
        lcl_valid = same * lcl_mask.unsqueeze(0) * lcl_mask.unsqueeze(1)
        time_diff = (lcl_ts.unsqueeze(0) - lcl_ts.unsqueeze(1))
        big_time = time_diff.max()*1e2
        mod_time_diff = time_diff.abs() + (big_time * ~same)
        closest_td, closest_idxs = mod_time_diff.topk(k, largest=False)
        final_valid[outer_idxs] = torch.gather(lcl_valid, 1, closest_idxs)
        final_idx[outer_idxs] = outer_idxs[1][closest_idxs]
        final_vec[outer_idxs] = closest_td
    final_vec = final_vec.unsqueeze(-1)
    return final_idx, final_vec, final_valid

def space(mask, ts, dist, adj, keys, points):
    expand_keys = keys.unsqueeze(2)
    key_ts = ts.unsqueeze(2)
    expand_points = points.unsqueeze(1)
    point_ts = ts.unsqueeze(1)
    # Separate spatial distances from time distance
    # Calculate spatial distances
    dist_vec = expand_keys - expand_points
    dist_vec = torch.cat((dist_vec, dist.unsqueeze(-1)), dim=-1)
    # Calculate mask
    # Units in a single timestep cannot be their own neighbor
    same_unit = torch.eye(dist.size(1), dtype=torch.bool, device=dist.device).unsqueeze(0)
    # Units must be in the same timestep to be neighbors
    same_t = (key_ts == point_ts)
    # The unit must not be disabled by the batch's time mask or unit mask
    # Combine the above
    valid = same_unit.logical_not() * same_t
    valid *= adj
    valid *= mask.unsqueeze(1) * mask.unsqueeze(2)
    return valid, dist_vec, dist

def time(time_enc, mask, ids, keys, points):
    expand_keys = keys.unsqueeze(2)
    key_ids = ids.unsqueeze(2)
    expand_points = points.unsqueeze(1)
    point_ids = ids.unsqueeze(1)
    # Separate spatial distances from time distance
    # Calculate spatial distances
    dist_vec = expand_keys - expand_points
    dist = dist_vec.abs()
    dist_vec = dist_vec.unsqueeze(-1)
    # Determine which distances matter
    # Units in a single timestep cannot be their own neighbor
    same_id = (key_ids == point_ids)
    # Combine the above
    masks = mask.unsqueeze(1) * mask.unsqueeze(2)
    valid = same_id * masks
    dist_vec = dist_vec.float()
    dist = dist.float()
    return valid, time_enc(dist_vec), dist

def query(time_enc, key_mask, point_mask, key_ids, point_ids, keys, points):
    key_ts = keys.unsqueeze(2)
    key_ids = key_ids.unsqueeze(2)
    point_ts = points.unsqueeze(1)
    point_ids = point_ids.unsqueeze(1)
    # Calculate spatial distances
    dist_vec = key_ts - point_ts
    dist = dist_vec.abs()
    dist_vec = dist_vec.unsqueeze(-1)
    # Determine which distances matter
    same_id = (key_ids == point_ids)
    # Combine the above
    valid = same_id * point_mask.unsqueeze(1) * key_mask.unsqueeze(2)
    dist_vec = dist_vec.float()
    dist = dist.float()
    return valid, time_enc(dist_vec), dist

