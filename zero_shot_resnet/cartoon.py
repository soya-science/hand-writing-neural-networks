"""Cartoon faces for probing the frozen backbone, built on v1's drawing code.

v1's synthetic_faces.draw() and sample_params() are reused unchanged. On top of them this
file adds:
- twin pairs: one sampled resting face drawn twice, at rest and with one facial action
  switched on, under the same pose, lighting and noise, so the pair differs only in that
  action;
- skin texture, gamma and mirror flips on every face, and a "rough" photo regime with bigger
  pose jitter, harsher lighting and blur. The probe prefers channels that behave the same in
  both regimes, i.e. that do not depend on the cartoon's clean rendering.

No real image is used anywhere in this file.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zero_shot",
                             "emotion_cnn"))
import synthetic_faces as SF  # noqa: E402  (v1, unchanged)

REGIMES = ("standard", "rough")

# One facial action per entry: the drawing parameters it sets (see v1's NEUTRAL for units).
ACTIONS = {
    "brow_raise":       dict(brow_raise=5.5, brow_arch=5, forehead=1),     # AU1+2
    "brows_inner_up":   dict(brow_tilt=5),                                 # AU1
    "brow_lower":       dict(brow_raise=-3, brow_tilt=-5, furrow=1),       # AU4
    "eyes_wide":        dict(eye_open=11),                                 # AU5
    "eyes_narrow":      dict(eye_open=4.5),                                # AU6/7
    "nose_wrinkle":     dict(nose_wrinkle=1),                              # AU9
    "upper_lip_raise":  dict(upper_lip=4),                                 # AU10
    "lip_corners_up":   dict(curve=6, folds=1.0),                          # AU12
    "lip_corners_down": dict(curve=-5),                                    # AU15
    "mouth_open":       dict(open=12),                                     # AU26/27
    "teeth":            dict(open=6, teeth=1),                             # AU25 + teeth
}


def _texture(rng, amp):
    """Smooth random blotches at three scales, like uneven skin and shadows."""
    a = np.zeros((100, 100))
    for cells, w in ((4, 1.0), (10, 0.6), (25, 0.35)):
        g = rng.normal(0, 1, (cells, cells)).astype(np.float32)
        a += w * np.asarray(Image.fromarray(g).resize((100, 100), Image.BICUBIC))
    return amp * a / 1.5


def _affine(img, rng, rot, scale, shift):
    ang, sc = np.deg2rad(rng.uniform(-rot, rot)), rng.uniform(1 - scale, 1 + scale)
    tx, ty = rng.uniform(-shift, shift, 2) * SF.S
    c = 50 * SF.S
    m00, m01 = np.cos(ang) / sc, np.sin(ang) / sc
    m10, m11 = -m01, m00
    coeffs = (m00, m01, c - m00 * (c + tx) - m01 * (c + ty),
              m10, m11, c - m10 * (c + tx) - m11 * (c + ty))
    return img.transform(img.size, Image.AFFINE, coeffs, Image.BICUBIC, fillcolor=SF.BG)


def _standard(img, rng):
    a = SF._photograph(SF._jitter_pose(img, rng), rng).astype(float)
    return a + _texture(rng, rng.uniform(0, 12))[..., None]


def _rough(img, rng):
    a = np.asarray(_affine(img, rng, 8, 0.1, 5).resize((100, 100), Image.LANCZOS), float)
    yy, xx = np.mgrid[:100, :100] / 99 - 0.5
    a = a + rng.uniform(-50, 50) * xx + rng.uniform(-30, 30) * yy
    a = (a - a.mean()) * rng.uniform(0.6, 1.4) + a.mean() + rng.uniform(-30, 30)
    blurred = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    a = np.asarray(blurred.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6))), float)
    a = a + _texture(rng, rng.uniform(5, 25)) + rng.normal(0, 6, a.shape)
    return np.repeat(a[..., None], 3, -1)


def render(p, seed, regime):
    """Draw parameters p as a (100, 100, 3) uint8 photo; the same seed gives the same pose,
    lighting and noise."""
    rng = np.random.default_rng(seed)
    a = (_standard if regime == "standard" else _rough)(SF.draw(p), rng)
    a = 255 * (np.clip(a, 0, 255) / 255) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.5:
        a = a[:, ::-1]
    return np.clip(a, 0, 255).astype(np.uint8)


def twins(action, n, seed, regime):
    """n resting faces and the same n faces with `action` switched on."""
    rng = np.random.default_rng(seed)
    rest, active = [], []
    for _ in range(n):
        p0 = SF.sample_params("neutral", rng)
        p1 = dict(p0)
        for k, v in ACTIONS[action].items():
            p1[k] = v if k == "teeth" else v * rng.uniform(0.75, 1.25) + rng.normal(0, SF.SPREAD.get(k, 0))
        p1["eye_open"] = max(p1["eye_open"], 2.5)
        s = int(rng.integers(1 << 30))
        rest.append(render(p0, s, regime))
        active.append(render(p1, s, regime))
    return np.stack(rest), np.stack(active)


def emotion_set(per_class, seed, regime):
    """v1's emotion prototypes (with their variants), rendered in the given regime."""
    rng = np.random.default_rng(seed)
    X = [render(SF.sample_params(c, rng), int(rng.integers(1 << 30)), regime)
         for c in SF.CLASSES for _ in range(per_class)]
    return np.stack(X), np.repeat(np.arange(len(SF.CLASSES)), per_class)
