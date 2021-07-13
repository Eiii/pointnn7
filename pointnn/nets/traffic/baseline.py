import torch

class Random:
    def __init__(self, range):
        self.range = range

    def get_args(self, item):
        return [item['tgt_id']]

    def __call__(self, ids):
        r = torch.rand_like(ids, dtype=torch.float)
        return (r*(self.range[1]-self.range[0]))+self.range[0]

class Nearest:
    def __init__(self, device='cuda'):
        self.device = device

    def get_args(self, item):
        xx = ['hist_data', 'hist_id', 'hist_t', 'tgt_id']
        return [item[x] for x in xx]

    def __call__(self, hist_data, hist_id, hist_t, tgt_id):
        out = []
        for id_ in tgt_id[0]:
            mask = (hist_id == id_.item())
            min_t = hist_t[mask].min()
            mask *= (hist_t == min_t)
            val = hist_data[mask].item()
            out.append(val)
        return torch.tensor(out).unsqueeze(0).to(self.device)

class Mean:
    def __init__(self, device='cuda'):
        self.device = device

    def get_args(self, item):
        xx = ['hist_data', 'hist_id', 'hist_t', 'tgt_id']
        return [item[x] for x in xx]

    def __call__(self, hist_data, hist_id, hist_t, tgt_id):
        out = []
        for id_ in tgt_id[0]:
            mask = (hist_id == id_.item())
            val = hist_data[mask].mean().item()
            out.append(val)
        return torch.tensor(out).unsqueeze(0).to(self.device)
