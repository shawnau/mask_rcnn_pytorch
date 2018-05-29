import random
from torch.utils.data.sampler import *


class ConstantSampler(Sampler):
    def __init__(self, data, list):
        self.num_samples = len(list)
        self.list = list

    def __iter__(self):
        return iter(self.list)

    def __len__(self):
        return self.num_samples


# see trorch/utils/data/sampler.py
class FixLengthRandomSampler(Sampler):
    def __init__(self, data, length=None):
        self.len_data = len(data)
        self.length = length or self.len_data

    def __iter__(self):
        l = []
        while 1:
            ll = list(range(self.len_data))
            random.shuffle(ll)
            l = l + ll
            if len(l) >= self.length: break

        l = l[:self.length]
        return iter(l)

    def __len__(self):
        return self.length
