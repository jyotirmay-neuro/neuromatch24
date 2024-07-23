
from typing import Callable
import torch.nn as nn


class Sensor(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 3,
        n_hidden: int = 1,
        feat_start_idx: int = 5,
        transform_features: Callable | None = None
    ) -> None:
        super().__init__()
        self.feat_start_idx = feat_start_idx
        self.feat_end_idx = feat_start_idx + input_dim
        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                    nn.ReLU(),
                ) for i in range(n_hidden)
            ],
            nn.Linear(hidden_dim, output_dim)
        )
        self.transform_features = transform_features

    def forward(self, observations):
        features = observations[..., self.feat_start_idx : self.feat_end_idx]
        if self.transform_features is not None:
            features = self.transform_features
        out =  self.layers(features)
        return out[...,0], out[...,1], out[...,2]
