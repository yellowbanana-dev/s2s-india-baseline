"""G1 patch-ViT smoke tests (task #6 guardrails). CPU only, no network."""
import torch
from omegaconf import OmegaConf

from s2s.models.patch_vit import PatchViT

_GRID = (32, 64)


def _make_model_cfg():
    return OmegaConf.create(
        {
            "patch_size": 2,
            "embed_dim": 256,
            "depth": 6,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "drop_rate": 0.1,
        }
    )


def test_output_shape():
    cfg = _make_model_cfg()
    in_channels, out_channels, lead = 6, 2, 6
    model = PatchViT(in_channels, out_channels, lead, cfg)
    x = torch.randn(3, in_channels, *_GRID)
    out = model(x)
    assert out.shape == (3, lead, out_channels, *_GRID)


def test_param_count_is_small():
    cfg = _make_model_cfg()
    model = PatchViT(in_channels=6, out_channels=2, lead=6, cfg=cfg)
    n_params = sum(p.numel() for p in model.parameters())
    # "Kept tiny on purpose" (module docstring) -- a few M params at embed_dim=256.
    assert 0 < n_params < 20_000_000


def test_gradients_flow():
    cfg = _make_model_cfg()
    in_channels, out_channels, lead = 6, 2, 6
    model = PatchViT(in_channels, out_channels, lead, cfg)
    x = torch.randn(2, in_channels, *_GRID, requires_grad=True)
    out = model(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
