import itertools
import numpy as np
from torch.utils.data.sampler import Sampler, BatchSampler, SubsetRandomSampler


class TwoStreamBatchSampler(BatchSampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.inter_batch_size = 3 * batch_size // 4
        self.batch_size = batch_size

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, 3*self.batch_size // 4),
                    grouper(secondary_iter,  self.batch_size // 4))
        )

    def __len__(self):
        return len(self.primary_indices) // self.inter_batch_size

class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        for i in self.indices:
            yield i

    def __len__(self):
        return len(self.indices)
    
def iterate_once(iterable):
   
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

