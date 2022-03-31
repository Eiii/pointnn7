import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import linalg

from .pointconv import calc_neighbor_info, _gather_neighbor_info
from ..utils import pairwise


class Spectral(nn.Module):
    def __init__(self, in_size, out_size, pos_size, eig_dims):
        super().__init__()
        self.out_size = out_size
        self.param = torch.empty(out_size, in_size+pos_size, eig_dims)
        nn.init.kaiming_uniform_(self.param, nonlinearity='relu')

    def forward(self, feats, pos, times, ids, eig):
        # feats - count x F
        # square_feats - times x nodes x F
        square_feats = get_square_feats(feats, pos, times, ids)
        out_feats = self.transform(square_feats, eig)
        out_feats = out_feats.contiguous()
        flat_feats = flatten(out_feats)
        return flat_feats

    def transform(self, feats, pos, eig):
        f = feats.permute(0, 1, 3, 2)
        if eig.size(0) != self.param.size(-1):
            missing = self.param.size(-1) - eig.size(0)
            z_size = (missing,) + eig.size()[1:]
            zs = torch.zeros(z_size)
            eig = torch.cat((eig, zs.to(eig.device)), dim=0)
        x1 = torch.matmul(f, eig.T)
        x1_exp = x1.unsqueeze(2)
        # For dynamic scenes there may be less nodes than the
        # eigenvector cutoff.
        # In this case, we should fill in the 'missing' eigenvector rows
        # with zeros. Otherwise the transformation will fail
        p_exp = self.param.view(*[1, 1, *self.param.size()]).to(x1.device)
        x2 = x1_exp * p_exp
        x3 = x2.sum(dim=3)
        x4 = torch.matmul(x3, eig)
        out_f = x4.permute(0, 1, 3, 2)
        return F.relu(out_f)

class SpectralStack(nn.Module):
    def __init__(self, in_size, out_size, pos_size, hidden, eig_dims, lap_type):
        super().__init__()
        sizes = [in_size, *hidden, out_size]
        self.lap = lap_type
        self.dist_thresh = 0.5
        self.layers = nn.ModuleList()
        self.eig_dims = eig_dims
        for in_, out_ in pairwise(sizes):
            s = Spectral(in_, out_, pos_size, eig_dims)
            self.layers.append(s)

    def forward(self, feats, pos, times, ids, dist_fn, dist_data, eig):
        if eig is not None:
            # Static structure, use precomputed eigenvectors
            # Also lets us assmue # of entities in each problem is constant
            x = get_square_feats(feats, pos, times, ids)
            for layer in self.layers:
                x = layer.transform(x, eig)
            return flatten(x.contiguous())
        else:
            f_out = self.dynamic_eigs(feats, pos, times, ids, dist_fn, dist_data, eig)
            return f_out

    def dynamic_eigs(self, feats, pos, times, ids, dist_fn, dist_data, eig):
        valid, dist_vec, dist = dist_fn(pos, pos)
        valid_ids = (ids != 0) # Hack
        # Calc eigs
        eigs = {}
        info = (feats, pos, dist, times, ids, valid_ids)
        for bid in range(feats.size(0)):
            f, p, d, t, i, vi = [x.select(0, bid) for x in info]
            for time, f, p, i, d, t_mask in split_times(t, f, p, d, i):
                if t_mask.sum() == 0:
                    continue
                eig = self.dynamic_eig(d).to(f.device)
                eigs[bid, time.item()] = eig
        for layer in self.layers:
            info = (feats, pos, dist, times, ids, valid_ids)
            # For each batch...
            next_feats = []
            for bid in range(feats.size(0)):
                f, p, d, t, i, vi = [x.select(0, bid) for x in info]
                batch_out = torch.zeros((feats.size(1), layer.out_size), device=feats.device)
                for time, f, p, i, d, t_mask in split_times(t, f, p, d, i):
                    if t_mask.sum() == 0:
                        continue
                    eig = eigs[bid, time.item()]
                    f = torch.cat((f, p), dim=-1)
                    f = f.unsqueeze(0).unsqueeze(0)
                    out_f = layer.transform(f, pos, eig).squeeze(0).squeeze(0)
                    batch_out[t_mask] = out_f
                next_feats.append(batch_out)
            next_feats = torch.stack(next_feats)
            feats = next_feats
        return feats

    def dynamic_eig(self, dists):
        edges = dists < self.dist_thresh
        lap_fn = {'graph': graph_laplacian, 'comb': comb_laplacian}[self.lap]
        lap = lap_fn(edges)
        spec = calc_spec(lap, self.eig_dims)
        return spec

def split_times(times, feats, pos, dists, ids):
    all_ts = times.unique()
    valid_ids = (ids != 0) # HACK
    for t in all_ts:
        t_mask = (times == t) * valid_ids
        t_feats = feats[t_mask]
        t_id = ids[t_mask]
        t_dists = dists[t_mask][:, t_mask]
        t_pos = pos[t_mask]
        yield t, t_feats, t_pos, t_id, t_dists, t_mask

def get_square_feats(feats, pos, times, ids):
    all_times = times.unique(dim=1)
    slices = []
    for t_idx in range(all_times.size(1)):
        t = all_times[:, t_idx]
        time_match = times == t.unsqueeze(-1)
        bs = feats.size(0)
        match_ids = ids[time_match].view(bs, -1)
        assert match_ids.size(0) == 1 or (match_ids[0] == match_ids[1]).all()
        match_fs = feats[time_match].view(bs, -1, feats.size(-1))
        slices.append(match_fs)
    square_feats = torch.stack(slices, dim=1)
    return square_feats

def flatten(square_feats):
    sz = [square_feats.size(0), -1, square_feats.size(-1)]
    return square_feats.view(*sz)


def graph_laplacian(adj):
    deg = adj.sum(dim=0)
    half_deg = deg.pow(-0.5)
    deg_mat = torch.diag(half_deg).to(adj.device)
    lap = torch.eye(adj.size(0), device=adj.device) - deg_mat.matmul(adj.float()).matmul(deg_mat)
    return lap

def comb_laplacian(adj):
    deg = adj.sum(dim=0)
    deg_mat = torch.diag(deg).to(adj.device)
    lap = deg_mat - adj.int()
    return lap.float()

def calc_spec(lap, cutoff):
    lap = lap.cpu()
    np_eigval, np_eig = linalg.eig(lap)
    eig_idx = np_eigval.argsort()[::-1]
    np_eigval = np_eigval[eig_idx]
    np_eig = np_eig[:, eig_idx]
    full_eig = torch.tensor(np_eig, device=lap.device)
    eig = full_eig.real if full_eig.dtype == torch.complex64 else full_eig
    return eig[:cutoff, :]
