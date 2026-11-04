"""Forward pass of the hand-wired emotion CNN, in plain NumPy.

All numbers come from weights.py; this file only turns those tables into arrays and runs
them. forward() returns every intermediate activation so each stage can be inspected.
"""
import numpy as np

import weights as W

CONV1_NAMES = list(W.CONV1)
CONV2_NAMES = list(W.CONV2)
AU_NAMES = list(W.AUS)


def _build_conv1():
    k = np.stack([s * np.array(g, float) for s, g, _ in W.CONV1.values()])[:, None]
    b = np.array([bias for _, _, bias in W.CONV1.values()])
    return k, b  # (8, 1, 5, 5), (8,)


def _build_conv2():
    k = np.zeros((len(CONV2_NAMES), len(CONV1_NAMES), 5, 5))
    b = np.zeros(len(CONV2_NAMES))
    for f, (scale, grids, bias) in enumerate(W.CONV2.values()):
        for name, g in grids.items():
            k[f, CONV1_NAMES.index(name)] = scale * np.array(g, float)
        b[f] = bias
    return k, b  # (10, 8, 5, 5), (10,)


def _build_zones(size=25):
    masks = {}
    for name, boxes in W.ZONES.items():
        m = np.zeros((size, size), bool)
        for r0, r1, c0, c1 in boxes:
            m[r0:r1 + 1, c0:c1 + 1] = True
        masks[name] = m
    return masks


K1, B1 = _build_conv1()
K2, B2 = _build_conv2()
ZONE_MASKS = _build_zones()
FC = np.array([W.FC[a] for a in AU_NAMES])  # (n_aus, 7)
FC_BIAS = np.array(W.FC_BIAS)


def conv2d(x, k, b):
    """'Same' cross-correlation. x: (N, C, H, W), k: (F, C, kh, kw), b: (F,)."""
    p = k.shape[-1] // 2
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    win = np.lib.stride_tricks.sliding_window_view(xp, k.shape[-2:], axis=(2, 3))
    return np.einsum("nchwij,fcij->nfhw", win, k, optimize=True) + b[None, :, None, None]


def relu(x):
    return np.maximum(x, 0.0)


def capped_relu(x):
    """ReLU clipped at 1, so 1 means "fully present" and one strong cue cannot pass an
    AND threshold on its own."""
    return np.clip(x, 0.0, 1.0)


def pool2(x, op):
    n, c, h, w = x.shape
    x = x[:, :, : h // 2 * 2, : w // 2 * 2].reshape(n, c, h // 2, 2, w // 2, 2)
    return op(x, axis=(3, 5))


def front_end(rgb):
    """rgb: (N, 100, 100, 3) in any scale -> standardized gray face (N, 1, 50, 50)."""
    gray = (rgb.astype(float) @ np.array(W.LUMA))[:, None]
    g50 = pool2(gray, np.mean)
    r0, r1, c0, c1 = W.STANDARDIZE_BOX
    box = g50[:, :, r0:r1 + 1, c0:c1 + 1]
    mu = box.mean(axis=(1, 2, 3), keepdims=True)
    sd = box.std(axis=(1, 2, 3), keepdims=True) + 1e-6
    return (g50 - mu) / sd


def parts(face50):
    """Standardized face (N, 1, 50, 50) -> conv1, pooled conv1 and conv2 activations."""
    c1 = capped_relu(conv2d(face50, K1, B1))
    p1 = pool2(c1, np.max)
    c2 = capped_relu(conv2d(p1, K2, B2))
    return c1, p1, c2


def action_units(p1, c2):
    """Zone-pool the 25x25 maps into the AU evidence vector (N, n_aus)."""
    maps = {n: p1[:, i] for i, n in enumerate(CONV1_NAMES)}
    maps.update({n: c2[:, i] for i, n in enumerate(CONV2_NAMES)})
    out = np.zeros((p1.shape[0], len(AU_NAMES)))
    for a, (bias, terms) in enumerate(W.AUS.values()):
        out[:, a] = bias
        for coef, op, map_name, zone in terms:
            inside = maps[map_name][:, ZONE_MASKS[zone]]
            out[:, a] += coef * (inside.max(axis=1) if op == "max" else inside.mean(axis=1))
    return relu(out)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(rgb):
    """rgb: (N, 100, 100, 3) -> dict of every activation, ending in 'probs' (N, 7)."""
    face = front_end(rgb)
    c1, p1, c2 = parts(face)
    au = action_units(p1, c2)
    logits = au @ FC + FC_BIAS
    return dict(face=face, conv1=c1, pool1=p1, conv2=c2, au=au, logits=logits,
                probs=softmax(logits / W.SOFTMAX_TEMPERATURE))


def predict(rgb, batch=256):
    """Class indices for (N, 100, 100, 3) images, in batches to bound memory."""
    return np.concatenate([forward(rgb[i:i + batch])["logits"].argmax(1)
                           for i in range(0, len(rgb), batch)])
