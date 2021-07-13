from .base import Problem
from ..data.starcraft import StarcraftDataset, parse_frame, collate#, collate_voxelize

import functools

import torch
import torch.nn.functional as F

class StarcraftScene(Problem):
    def __init__(self, data_path, max_hist=1, num_pred=1, max_files=None,
                 hist_dist=None, pred_dist=None, hist_dist_args=None, pred_dist_args=None,
                 voxel_res=None, frame_skip=2):
        # Load training & test dataset
        args = {'max_files': max_files, 'max_hist': max_hist, 'num_pred': num_pred,
                'hist_dist': hist_dist, 'pred_dist': pred_dist,
                'hist_dist_args': hist_dist_args, 'pred_dist_args': pred_dist_args,
                'frame_skip': frame_skip}
        self.train_dataset = StarcraftDataset(data_path, **args)
        self.valid_dataset = StarcraftDataset(data_path+'/test', **args)
        if voxel_res is None:
            self.collate_fn = collate
        else:
            vres = [voxel_res, voxel_res, 1]
            #self.collate_fn = functools.partial(collate_voxelize, vres)

    def loss(self, item, pred):
        l = sc2_frame_loss(item, pred)
        avg_loss = l.sum() / unit_count(item)
        return avg_loss

def unit_count(item):
    return (item['pred_ts_mask'].unsqueeze(-1)*item['pred_ids_mask'].unsqueeze(-2)).sum()

def sc2_frame_loss(item, pred, reduction='sum'):
    target = parse_frame(item['pred_data'])
    # Health loss
    h_loss = F.mse_loss(pred['health'], target['health'], reduction='none').squeeze(-1)
    # Shield loss
    s_loss = F.mse_loss(pred['shields'], target['shields'], reduction='none').squeeze(-1)
    # Position loss
    p_loss = ((target['pos'] - pred['pos'])**2).sum(dim=-1)
    # Orientation loss
    _, idx_target = target['ori'].max(dim=-1)
    o_loss = F.cross_entropy(pred['ori'].permute(0, -1, 1, 2), idx_target,
                             reduction='none', ignore_index=-1)
    # Alive loss
    alive_target = target['alive'].squeeze(-1).long()
    a_loss = F.cross_entropy(pred['alive'].permute(0, -1, 1, 2), alive_target,
                             reduction='none')
    # Figure out which parts of the loss we should ignore/consider
    unit_pred_loss = torch.stack((p_loss, h_loss, s_loss, o_loss), dim=-1)
    if reduction == 'sum':
        alive_mask = target['alive'].bool().squeeze(-1)
        unit_pred_loss = unit_pred_loss.sum(dim=-1) * alive_mask
        total_loss = unit_pred_loss+a_loss
        total_loss *= item['pred_ts_mask'].unsqueeze(-1) * item['pred_ids_mask'].unsqueeze(-2)
        return total_loss
    elif reduction == 'none':
        alive_mask = target['alive'].bool()
        valid_mask = item['pred_ts_mask'].unsqueeze(-1) * item['pred_ids_mask'].unsqueeze(-2)
        alive_losses = torch.masked_select(a_loss, valid_mask)
        lss = []
        for i in range(unit_pred_loss.size(-1)):
            tmp_mask = valid_mask.unsqueeze(-1).expand(-1, -1, -1, unit_pred_loss.size(-1)).contiguous()
            tmp_sel = torch.zeros(1, 1, 1, unit_pred_loss.size(-1), dtype=torch.bool, device=tmp_mask.device)
            tmp_sel[:, :, :, i] = True
            tmp_mask *= tmp_sel
            lss.append(torch.masked_select(unit_pred_loss, tmp_mask))
        return torch.stack(lss+[alive_losses], dim=-1)
