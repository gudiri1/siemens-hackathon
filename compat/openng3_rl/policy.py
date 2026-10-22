"""Feature extractor with the same architecture as the organiser's
``HCMR-EV-RL/src/openng3_rl/policy.py`` (image CNN -> 256 features, state MLP -> 128 features).

The layer layout must stay identical so that saved weights load in both code bases.
"""

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class FluoroStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict) -> None:
        image_space = observation_space.spaces["image"]
        state_space = observation_space.spaces["state"]
        self.image_features = 256
        self.state_features = 128
        super().__init__(observation_space, self.image_features + self.state_features)
        channels = image_space.shape[0]
        cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flattened = cnn(torch.zeros(1, *image_space.shape)).shape[1]
        self.image_net = nn.Sequential(cnn, nn.Linear(flattened, self.image_features), nn.ReLU())
        self.state_net = nn.Sequential(
            nn.Linear(state_space.shape[0], self.state_features),
            nn.ReLU(),
            nn.Linear(self.state_features, self.state_features),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            (self.image_net(observations["image"]), self.state_net(observations["state"])),
            dim=1,
        )
