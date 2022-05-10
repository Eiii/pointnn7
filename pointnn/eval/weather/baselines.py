import torch
#from sklearn.neighbors import KNeighborsTransformer

import numpy as np
import math

RELH_IDX=0
TAIR_IDX=1
WSPD_IDX=2
PRES_IDX=8


def split_tensor_dict(d):
    it_d = {k:(x for x in v) for k,v in d.items()}
    while True:
        yield {k:next(v) for k,v in it_d.items()}


class SelfEnsemble:
    def __init__(self, net, count, kw={}):
        self.net = net
        self.count = count
        self.kw = kw

    def get_args(self, item):
        return self.net.get_args(item)

    def __call__(self, *args):
        preds = [self.net(*args, **self.kw) for _ in range(self.count)]
        return torch.stack(preds).mean(dim=0)


class KNN:
    def __init__(self, k, weight=True):
        self.k = k
        self.weight = weight

    def get_args(self, item):
        return (item,)

    def __call__(self, item):
        preds = [self.single(i) for i in split_tensor_dict(item)]
        return torch.stack(preds)

    def single(self, item):
        sel = (item['times']==0)&item['hist_mask']
        cols = [RELH_IDX, TAIR_IDX, WSPD_IDX, PRES_IDX]
        latest_hist = item['hist'][ sel]
        latest_hist = latest_hist[:, cols]
        latest_histq = item['hist_q'][sel]
        latest_histq = latest_histq[:, cols]
        latest_idxs = item['station_idxs'][sel]
        latest_pos = item['station_pos'][latest_idxs]
        target_pos = item['target_pos']
        knn = KNeighborsTransformer(n_neighbors=min(self.k, latest_hist.size(0)))
        knn.fit(latest_pos)
        n_dists, n_idxs = knn.kneighbors(target_pos)
        n_idxs = torch.tensor(n_idxs)
        closest_hist = latest_hist[n_idxs]
        if self.weight:
            n_dists = torch.tensor(n_dists)
            n_weights = 1/(n_dists**2)
            n_weights = n_weights/torch.sum(n_weights, dim=1, keepdim=True)
            closest_hist *= n_weights.unsqueeze(-1)
            result = closest_hist.sum(dim=1)
        else:
            result = closest_hist.mean(dim=1)
        return result
