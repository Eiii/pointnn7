from . import pointconv
from .base import Network
from .encodings import DirectEncoding, PeriodEncoding

import math

import torch
import torch.nn as nn
from functools import partial
from kaolin.models.PointNet import PointNetFeatureExtractor as Pointnet
from .hyperpointnet import HyperPointNet, PointNet as LocalPointNet

class ThorTest(Network):
    def __init__(self, time_encode='direct', ignore_scene=False):
        super().__init__()
        if time_encode == 'direct':
            self.time_encode = DirectEncoding()
        elif time_encode == 'period':
            self.time_encode = PeriodEncoding(10, 30)
        self.ignore_scene = ignore_scene
        latent_sizes = [2**6]*3
        pred_sz = 2**7
        self.space_dim = 3
        self.train_margins = [0, 1, 3, 7]
        self.make_encoder(latent_sizes)
        self.make_decoder(latent_sizes[-1], pred_sz)

    def make_encoder(self, latent_sizes):
        self.scene_convs = nn.ModuleList()
        self.time_convs = nn.ModuleList()
        self.combine_mlps = nn.ModuleList()
        weight_hidden = [2**6]*2
        c_mid = 32
        final_hidden = [2**7]*2
        default_args = {'weight_hidden': weight_hidden, 'c_mid': c_mid,
                        'final_hidden': final_hidden, 'norm': True,
                        'norm_type': 'batch', 'residual': False,
                        'e_reduce': None}
        combine_hidden = [2**8]*2
        in_size = self.space_dim
        for latent_sz in latent_sizes:
            args = dict(default_args)
            args.update({'neighbors': 256, 'c_in': 1, 'c_out': latent_sz,
                         'dim': self.space_dim+1})
            pc = pointconv.PointConv(**args)
            self.scene_convs.append(pc)
            args = dict(default_args)
            args.update({'neighbors': -1, 'c_in': in_size+latent_sz, 'c_out': latent_sz,
                         'dim': self.time_encode.out_dim})
            pc = pointconv.PointConv(**args)
            self.time_convs.append(pc)
            mlp_args = {'in_channels': in_size+2*latent_sz, 'feat_size': latent_sz,
                        'layer_dims': combine_hidden, 'output_feat': 'pointwise',
                        'norm': True, 'norm_type': 'batch', 'residual': False}
            pn = Pointnet(**mlp_args)
            self.combine_mlps.append(pn)
            in_size = latent_sz

    def make_decoder(self, latent_sz, pred_sz):
        weight_hidden = [2**8]*3
        c_mid = 64
        final_hidden = [2**8]*3
        pred_hidden = [2**8]*3
        args = {'weight_hidden': weight_hidden, 'c_mid': c_mid,
                'final_hidden': final_hidden, 'norm': True,
                'norm_type': 'batch', 'residual': False,
                'e_reduce': None,
                'neighbors': -1, 'c_in': latent_sz, 'c_out': pred_sz,
                'dim': self.time_encode.out_dim}
        self.pred_pc = pointconv.PointConv(**args)
        mlp_args = {'in_channels': pred_sz, 'feat_size': self.space_dim,
                    'layer_dims': pred_hidden, 'output_feat': 'pointwise',
                    'norm': True, 'norm_type': 'batch', 'residual': False}
        self.predict_mlp = Pointnet(**mlp_args)

    def encode(self, mask, obj_idxs, times, pos,
               scene_pts, scene_times, scene_idxs, scene_mask):
        scene_dist_fn = partial(scene_dist, mask, obj_idxs, times, scene_mask, scene_times, scene_idxs)
        time_dist_fn = partial(time_dist, self.time_encode, mask, obj_idxs)
        feats = pos
        for scene_pc, time_pc, combine_mlp in zip(self.scene_convs, self.time_convs, self.combine_mlps):
            scene_const_feats = torch.ones(scene_pts.shape[:-1]+(1,), device=scene_pts.device, dtype=scene_pts.dtype)
            scene_nei = scene_pc(pos, scene_pts, scene_const_feats, scene_dist_fn)
            if self.ignore_scene:
                scene_nei = torch.zeros_like(scene_nei)
            time_in = torch.cat([feats, scene_nei], dim=2)
            time_nei = time_pc(times, times, time_in, time_dist_fn)
            combine_in = torch.cat([feats, scene_nei, time_nei], dim=2)
            feats = combine_mlp(combine_in).transpose(1, 2)
        return feats

    def decode(self, mask, feats, obj_idx, times, target_mask, target_idx, target_times, margin_list=None):
        if margin_list is None:
            margin_list = self.train_margins
        # Add margins
        target_mask_exp = target_mask.unsqueeze(-1)
        target_idx_exp = target_idx.unsqueeze(-1)
        target_times_exp = target_times.unsqueeze(-1)
        margin_exp = torch.tensor(margin_list, device=target_mask_exp.device).view(1, 1, -1)
        target_mask_exp, target_idx_exp, target_times_exp, margin_exp = \
            torch.broadcast_tensors(target_mask_exp, target_idx_exp, target_times_exp, margin_exp)
        target_mask_exp, target_idx_exp, target_times_exp, margin_exp = \
            map(combine_last_dim, (target_mask_exp, target_idx_exp, target_times_exp, margin_exp))
        pred_dist_fn = partial(pred_dist, self.time_encode, margin_exp, target_mask_exp, mask, target_idx_exp, obj_idx)
        pred_feats = self.pred_pc(target_times_exp, times, feats, pred_dist_fn)
        pred = self.predict_mlp(pred_feats).transpose(1, 2)
        pred = restore_dim(pred, len(margin_list))
        return pred

    def forward(self, mask, obj_idxs, times, pos,
                target_mask, target_idxs, target_times,
                scene_pts, scene_times, scene_idxs, scene_mask,
                margin_list=None):
        encoded_feats = self.encode(mask, obj_idxs, times, pos, scene_pts, scene_times, scene_idxs, scene_mask)
        pred = self.decode(mask, encoded_feats, obj_idxs, times,
                           target_mask, target_idxs, target_times, margin_list)
        return pred

    def get_args(self, item):
        mask = item['mask']
        obj_idxs = item['obj_idx']
        times = item['time']
        pos = item['pos']
        target_idxs = obj_idxs
        target_times = times
        target_mask = mask
        scene_pts = item['scene_pts']
        scene_times = item['scene_times']
        scene_idxs = item['scene_idxs']
        scene_mask = item['scene_mask']
        return (mask, obj_idxs, times, pos,
                target_mask, target_idxs, target_times,
                scene_pts, scene_times, scene_idxs, scene_mask)

# Distance functions
def scene_dist(obj_mask, obj_idxs, obj_times, scene_mask, scene_times, scene_idxs, keys, points):
    # Determine validity -
    # Only where both are valid according to the mask
    #  and both are the same object
    #  and point.time < key.time
    obj_idxs_exp = obj_idxs.unsqueeze(-1)
    scene_idxs_exp = scene_idxs.unsqueeze(-2)
    diff_obj = (obj_idxs_exp != scene_idxs_exp).unsqueeze(-1).float()
    obj_times_exp = obj_times.unsqueeze(-1)
    scene_times_exp = scene_times.unsqueeze(-2)
    same_time = (obj_times_exp == scene_times_exp)
    valid = obj_mask.unsqueeze(-1) * scene_mask.unsqueeze(-2) * same_time
    dist_vec = (keys.unsqueeze(-2) - points.unsqueeze(-3))
    final_dist_vec = torch.cat([dist_vec, diff_obj], dim=-1)
    sqr_dist = (dist_vec**2).sum(-1)
    return valid, final_dist_vec, sqr_dist

def time_dist(time_encode, mask, obj_idxs, keys, points):
    # Determine validity -
    # Only where both are valid according to the mask
    #  and both are the same object
    #  and point.time < key.time
    idxs1 = obj_idxs.unsqueeze(-1)
    idxs2 = obj_idxs.unsqueeze(-2)
    time1 = keys.unsqueeze(-1)
    time2 = points.unsqueeze(-2)
    same_obj = (idxs1 == idxs2)
    point_before = time2 < time1
    mask_exp = mask.unsqueeze(-1)*mask.unsqueeze(-2)
    valid = mask_exp * same_obj * point_before
    # Calculate square distance in time
    # Unused as long as neighbors==-1, but may be relevant later
    sqr_dist = (time2-time1).float()**2
    # Calculate time distance vector
    # with periodic stuff
    time_diff = (time2-time1).float()
    dist_vec = time_encode.encode(time_diff)
    return valid, dist_vec, sqr_dist

def pred_dist(time_encode, margin, target_mask, mask, target_idxs, obj_idxs, keys, points):
    # Determine validity -
    # Only where both are valid according to the mask
    #  and both are the same object
    #  and point.time < key.time
    idxs1 = target_idxs.unsqueeze(-1)
    idxs2 = obj_idxs.unsqueeze(-2)
    time1 = keys.unsqueeze(-1)
    margin = margin.unsqueeze(-1)
    time2 = points.unsqueeze(-2)
    same_obj = (idxs1 == idxs2)
    point_before = time2 <= (time1-margin)
    mask_exp = target_mask.unsqueeze(-1)*mask.unsqueeze(-2)
    valid = mask_exp * same_obj * point_before
    # Calculate square distance in time
    # Unused as long as neighbors==-1, but may be relevant later
    sqr_dist = (time2-time1).float()**2
    # Calculate time distance vector
    # with periodic stuff
    time_diff = (time2-time1).float()
    dist_vec = time_encode.encode(time_diff)
    return valid, dist_vec, sqr_dist

def combine_last_dim(t):
    goal_size = t.size()[:-2]+(-1,)
    return t.contiguous().view(*goal_size)

def restore_dim(t, size):
    goal_size = (t.size(0), -1, size, t.size(-1))
    return t.view(*goal_size)
