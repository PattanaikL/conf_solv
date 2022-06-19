from typing import List, Optional, Type, Tuple,  Union
import numpy as np
import torch
import torch.jit
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TorchStandardScaler(nn.Module):
    def fit(self, x):
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, unbiased=False, keepdim=True)

    def transform(self, x):
        x = x - self.mean
        x = x / (self.std + 1e-7)
        return x

    def inverse_transform(self,x):
        x = x * (self.std + 1e-7)
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
        
        self.max = tensor.max(dim=0, keepdim=True)[0]
        self.min = tensor.min(dim=0, keepdim=True)[0]
        self.dist = self.max - self.min
        self.dist[self.dist == 0.0] = 1.0
        self.scale = 1.0 / self.dist
    def transform(self,tensor):  
        a, b = self.feature_range 
        tensor = tensor.sub(tensor.min(dim=0, keepdim=True)[0]).mul(self.scale)
        tensor = tensor.mul(b - a).add(a)
        return tensor
    def inverse_transform(self,tensor):
        a, b = self.feature_range 
        tensor = tensor.sub(a).div(b - a)
        tensor = tensor.div(self.scale).add(tensor.min(dim=0, keepdim=True)[0])  
        tensor = tensor.add(self.min[0])     
        return tensor

