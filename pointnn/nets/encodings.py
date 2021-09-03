import torch
import math

class DirectEncoding:
    out_dim = 1
    @staticmethod
    def encode(val):
        return val

class PeriodEncoding:
    def __init__(self, count, max):
        self.count = count
        self.out_dim = count*2
        self.max = max

    def encode(self, val):
        # TODO: This is dumb, too specific
        denom = torch.arange(self.count, dtype=torch.float, device=val.device)
        denom = self.max**(denom/(self.count-1))
        denom = denom.view(1, 1, 1, -1)
        frac = (math.pi/2)*val/denom
        result = torch.cat([torch.sin(frac), torch.cos(frac)], dim=-1)
        return result

def main():
    e = PeriodEncoding(2, 10)
    for x in range(11):
        t = torch.tensor(x)
        print(x, e.encode(t))

if __name__ == '__main__':
    main()
