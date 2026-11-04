"""Forward pass: frozen ResNet-50 layer3 -> hand-typed head -> emotion logits.

All head numbers come from head_weights.py; this file only turns those tables into arrays and
runs them.
"""
import numpy as np

import backbone as BB
import head_weights as HW

AU_NAMES = list(HW.AUS)
FC = np.array([HW.FC[a] for a in AU_NAMES])  # (n_aus, 7)
FC_BIAS = np.array(HW.FC_BIAS)


def _zone_mask(name):
    m = np.zeros((BB.GRID, BB.GRID), bool)
    for r0, r1, c0, c1 in HW.ZONES[name]:
        m[r0:r1 + 1, c0:c1 + 1] = True
    return m


ZONE_MASKS = {z: _zone_mask(z) for z in HW.ZONES}


def evidence(maps):
    """layer3 maps (N, 1024, 14, 14) -> raw AU evidence (N, n_aus), ~0 at rest, ~1 fully on."""
    out = np.zeros((len(maps), len(AU_NAMES)))
    for a, (zone, op, channels) in enumerate(HW.AUS.values()):
        idx = [c for c, _, _ in channels]
        rest = np.array([r for _, r, _ in channels])
        gain = np.array([g for _, _, g in channels])
        inside = maps[:, idx][:, :, ZONE_MASKS[zone]]
        v = inside.max(-1) if op == "max" else inside.mean(-1)
        out[:, a] = ((v - rest) / gain).mean(1)
    return out


def action_units(ev):
    return np.clip((ev - HW.AU_THRESHOLD) / (1 - HW.AU_THRESHOLD), 0.0, 1.0)


def forward(rgb):
    """rgb: (N, 100, 100, 3) uint8 -> dict with evidence, au, logits."""
    ev = evidence(BB.layer3(rgb))
    au = action_units(ev)
    return dict(evidence=ev, au=au, logits=au @ FC + FC_BIAS)


def predict(rgb, batch=256):
    return np.concatenate([forward(rgb[i:i + batch])["logits"].argmax(1)
                           for i in range(0, len(rgb), batch)])
