"""Forward pass: frozen ResNet-50 layer3 -> few-shot head -> emotion logits.

All head numbers come from few_shot_weights.py; this file only turns those tables into arrays
and runs them.
"""
import os
import sys

import numpy as np

import few_shot_weights as FW

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zero_shot_resnet"))
import backbone as BB  # noqa: E402

AU_NAMES = list(FW.AUS)
FC = np.array([FW.FC[a] for a in AU_NAMES])  # (n_aus, 7)
FC_BIAS = np.array(FW.FC_BIAS)


def _zone_mask(name):
    m = np.zeros((BB.GRID, BB.GRID), bool)
    for r0, r1, c0, c1 in FW.ZONES[name]:
        m[r0:r1 + 1, c0:c1 + 1] = True
    return m


ZONE_MASKS = {z: _zone_mask(z) for z in FW.ZONES}


def evidence(maps):
    """layer3 maps (N, 1024, 14, 14) -> AU evidence (N, n_aus), ~0 without the action, ~1 with."""
    out = np.zeros((len(maps), len(AU_NAMES)))
    for a, (zone, op, channels) in enumerate(FW.AUS.values()):
        idx = [c for c, _, _ in channels]
        rest = np.array([r for _, r, _ in channels])
        gain = np.array([g for _, _, g in channels])
        inside = maps[:, idx][:, :, ZONE_MASKS[zone]]
        v = inside.max(-1) if op == "max" else inside.mean(-1)
        out[:, a] = ((v - rest) / gain).mean(1)
    return out


def forward(rgb):
    """rgb: (N, 100, 100, 3) uint8 -> dict with evidence, au, logits."""
    ev = evidence(BB.layer3(rgb))
    au = np.clip(ev, 0.0, 1.0)
    return dict(evidence=ev, au=au, logits=au @ FC + FC_BIAS)
