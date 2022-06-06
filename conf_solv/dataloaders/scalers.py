from typing import List, Optional, Type, Tuple,  Union
import numpy as np
import torch
import torch.jit
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TorchStandardScaler(nn.Module):
    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x
    def inverse_transform(self,x):
        x *= (self.std + 1e-7)
        x += self.mean
        return x

class TorchMinMaxScaler(nn.Module):
    """MinMax Scaler

    Transforms each channel to the range [a, b].

    Parameters
    ----------
    feature_range : tuple
        Desired range of transformed data.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, tensor):
        """Fit features

        Parameters
        ----------
        stacked_features : tuple, list
            List of stacked features.

        Returns
        -------
        tensor 
            A tensor with scaled features using requested preprocessor.
        """

        tensor = torch.stack(tensor)

        # Feature range
        a, b = self.feature_range

        dist = tensor.max(dim=0, keepdim=True)[0] - tensor.min(dim=0, keepdim=True)[0]
        dist[dist == 0.0] = 1.0
        scale = 1.0 / dist
        tensor.mul_(scale).sub_(tensor.min(dim=0, keepdim=True)[0])
        tensor.mul_(b - a).add_(a)

        return tensor

#Let's do some testing: 
# data = torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]]).float()
# foo = TorchStandardScaler()
# foo.fit(data)
# print(foo.transform(data))
# data_np = data.numpy()
# scaler = StandardScaler()
# arr_norm = scaler.fit_transform(data_np)
# print(arr_norm)
