# Vendored from: Zhdanov, Lucic, Welling, van de Meent — Mosaic (ICML 2026)
# Original: https://github.com/maxxxzdn/mosaic  License: CC-BY-NC-4.0
# LOCAL MODIFICATION: dataset.WeatherMetadata import replaced with a forward-
# declaration stub.  We use Transformer directly (via MosaicBackbone adapter)
# and do not use WeatherModel in this project.
"""
WeatherModel wrapper (reference only — not used by s2s-india-baseline).

s2s-india-baseline drives `Transformer` directly via `MosaicBackbone` in
src/s2s/models/mosaic_backbone.py.  This file is retained for completeness and
attribution.
"""

import torch
from torch import nn


class _WeatherMetadataStub:
    """Placeholder type for the vendored WeatherModel signature."""
    static_data: torch.Tensor
    longitude: torch.Tensor
    latitude: torch.Tensor


class WeatherModel(nn.Module):
    """Weather forecasting model wrapper (upstream reference, not used locally)."""

    def __init__(self, model: nn.Module, weather_metadata: _WeatherMetadataStub):
        super().__init__()
        self.model = model
        self.model.initialize_static_vars(weather_metadata.static_data, weather_metadata.longitude, weather_metadata.latitude)
        self.model.initialize_interpolation(weather_metadata.longitude, weather_metadata.latitude)
        self.weather_metadata = weather_metadata

    def forward(self, norm_state: torch.Tensor, day_year_time: torch.Tensor, num_noise_samples: int):
        return self.model(norm_state, day_year_time, num_noise_samples)
