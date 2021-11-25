import numpy as np
import torch
from torch import Tensor


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def all_same(array: np.ndarray):
    """
    Checks whether all elements in the given array have the same value
    """
    return (array.flat[0] == array.flat).all()


def nanmean(v, nan_mask, inplace=True, **kwargs):
    """
    https://github.com/pytorch/pytorch/issues/21987#issuecomment-539402619
    """
    if not inplace:
        v = v.clone()
    v[nan_mask] = 0
    return v.sum(**kwargs) / (~nan_mask).float().sum(**kwargs)


def split_list(_list, sep):
    result, sub = [], []
    for x in _list:
        if x == sep:
            if sub:
                result.append(sub)
                sub = []
        else:
            sub.append(x)
    if sub:
        result.append(sub)
    return result


def limit(tensor: Tensor, max=10) -> Tensor:
    """
    Clamp tensor below specified limit. Useful for preventing unstable training when using logarithmic network outputs.
    """
    return torch.clamp(tensor, max=max)


class RunningStats:
    """Efficiently keeps track of mean and standard deviation of a set of observations"""

    def __init__(self, shape: tuple):
        if not isinstance(shape, tuple):
            raise ValueError('shape must be tuple')
        self.shape = shape
        self.num_seen = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.mean_of_squared = np.zeros(shape, dtype=np.float64)

    def add(self, data):
        assert data.shape[1:] == self.shape
        # cast to float64 to prevent overflows in next lines
        data = data.astype(np.float64)
        self.mean = (self.num_seen * self.mean + data.sum(0)) / (self.num_seen + data.shape[0])
        self.mean_of_squared = (self.num_seen * self.mean_of_squared + (data ** 2).sum(0)) / (self.num_seen + data.shape[0])
        self.num_seen += data.shape[0]

    @property
    def variance(self):
        return self.mean_of_squared - self.mean**2

    @property
    def std(self):
        return np.sqrt(self.variance)
