from typing import List, Optional, Type, Tuple,  Union
import numpy as np
import torch
import torch.jit
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TorchStandardScaler(nn.Module):
    def fit(self, x):
        mean = x.mean()
        std = x.std()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def transform(self, x):
        x = x - self.mean
        x = x / (self.std + 1e-11)
        return x

    def inverse_transform(self, x):
        x = x * (self.std + 1e-11)
        x = x + self.mean
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
        super().__init__()
        self.__dict__.update(kwargs)

    def fit(self, tensor):
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

        #tensor = torch.stack(tensor)

        # Feature range
        
        max = tensor.max()
        min = tensor.min()
        dist = max - min
        dist[dist == 0.0] = 1.0
        scale = 1.0 / dist

        self.register_buffer('min', min)
        self.register_buffer('scale', scale)

    def transform(self, tensor):
        a, b = self.feature_range 
        tensor = tensor.sub(self.min).mul(self.scale)
        tensor = tensor.mul(b - a).add(a)
        return tensor

    def inverse_transform(self, tensor):
        a, b = self.feature_range 
        tensor = tensor.sub(a).div(b - a)
        tensor = tensor.div(self.scale).add(self.min)
        return tensor

