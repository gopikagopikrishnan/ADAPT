"""
inference/fuse_weights.py

Post-training parameter-space weight fusion for ADAPT.

Given three independently trained models θ_DAS, θ_FDMAS, θ_Capon, the fused
model is:

    θ_fused = α·θ_DAS + β·θ_FDMAS + γ·θ_Capon    (α + β + γ = 1)

This linear combination produces a model that interpolates the
beamforming characteristics of each algorithm without re-training,
giving clinicians a tunable knob between resolution, contrast, and
speckle reduction.

Usage:-
CLI — equal-weight fusion (default):
    python inference/fuse_weights.py \\
        --das   Results/chkpt/das/iter_1/best.pt \\
        --fdmas Results/chkpt/fdmas/iter_1/best.pt \\
        --capon Results/chkpt/capon/iter_1/best.pt \\
        --out   Results/chkpt/fused.pt

CLI - custom weights:
    python inference/fuse_weights.py ... --alpha 0.5 --beta 0.3 --gamma 0.2

Python API:
    from inference.fuse_weights import fuse_models
    model = fuse_models(path_das, path_fdmas, path_capon, alpha=0.5, beta=0.3, gamma=0.2)
    torch.save(model.state_dict(), "fused.pt")
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch

from model.adapt_model import FixedUNetBeamformer


def fuse_models(
    path_das:   str,
    path_fdmas: str,
    path_capon: str,
    alpha:      float = 1 / 3,
    beta:       float = 1 / 3,
    gamma:      float = 1 / 3,
    device:     str   = "cpu",
) -> FixedUNetBeamformer:
    """Create a fused FixedUNetBeamformer from three task-specific checkpoints.

    Parameters
    ----------
    path_das / path_fdmas / path_capon :
        Paths to ``best.pt`` (state-dict only) from each task trainer.
    alpha, beta, gamma :
        Fusion coefficients.  Must satisfy α + β + γ ≈ 1.
    device :
        Device on which to load and fuse weights.

    Returns

    FixedUNetBeamformer with fused weights, on CPU.
    """
    total = alpha + beta + gamma
    if abs(total - 1.0) > 1e-4:
        raise ValueError(
            f"Fusion coefficients must sum to 1, got α={alpha}+β={beta}+γ={gamma}={total:.4f}"
        )

    def _load(path: str) -> dict[str, torch.Tensor]:
        ckpt = torch.load(path, map_location=device)
        # Trainer saves best.pt as raw state_dict; model.pt wraps in a dict.
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        return ckpt

    sd_das   = _load(path_das)
    sd_fdmas = _load(path_fdmas)
    sd_capon = _load(path_capon)

    # Validate all three have identical keys
    assert sd_das.keys() == sd_fdmas.keys() == sd_capon.keys(), (
        "State-dict keys do not match — ensure all three checkpoints come from "
        "the same FixedUNetBeamformer architecture."
    )

    # Linear combination
    fused_sd = {
        key: alpha * sd_das[key] + beta * sd_fdmas[key] + gamma * sd_capon[key]
        for key in sd_das
    }

    model = FixedUNetBeamformer()
    model.load_state_dict(fused_sd)
    model.eval()

    print(
        f"[fuse_weights] Fused model created.\n"
        f"  α (DAS)={alpha:.3f}  β (FDMAS)={beta:.3f}  γ (Capon)={gamma:.3f}"
    )
    return model


def run_inference(
    model:          FixedUNetBeamformer,
    tofc_tensor:    torch.Tensor,          # (B, 128, H, W)
    device:         str = "cpu",
) -> torch.Tensor:
    """Apply fused model to a ToFC input and return softmax apodization weights.

    Parameters

    model :
        Fused model from ``fuse_models``.
    tofc_tensor :
        Input tensor (B, N_ELEMENTS, H, W).
    device :
        Inference device.

    Returns

    Apodization weights (B, N_ELEMENTS, H, W).
    """
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x = tofc_tensor.to(device)
        return model(x)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAPT weight fusion")
    parser.add_argument("--das",   required=True, help="Path to DAS best.pt")
    parser.add_argument("--fdmas", required=True, help="Path to FDMAS best.pt")
    parser.add_argument("--capon", required=True, help="Path to Capon best.pt")
    parser.add_argument("--out",   required=True, help="Output path for fused.pt")
    parser.add_argument("--alpha", type=float, default=1/3,
                        help="DAS weight (default 1/3)") # Tunable
    parser.add_argument("--beta",  type=float, default=1/3,
                        help="FDMAS weight (default 1/3)") # Tunable
    parser.add_argument("--gamma", type=float, default=1/3,
                        help="Capon weight (default 1/3)") # Tunable

    a = parser.parse_args()
    model = fuse_models(
        a.das, a.fdmas, a.capon,
        alpha=a.alpha, beta=a.beta, gamma=a.gamma,
    )
    torch.save(model.state_dict(), a.out)
    print(f"[fuse_weights] Saved fused model → {a.out}")
