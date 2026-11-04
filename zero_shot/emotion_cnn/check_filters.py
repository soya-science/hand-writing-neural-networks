"""Unit checks for the hand-typed filters, using ideal patterns drawn from scratch.

1. conv1: the linear response of each filter to each ideal pattern at unit contrast.
   The diagonal should read 1.0 and the rest should be small.
2. conv2: simple shapes drawn at face scale, passed through conv1 -> pool -> conv2.
   Each shape should light up its own part detector the most.
"""
import numpy as np
from PIL import Image, ImageDraw

import model as M

S = 4  # supersampling for anti-aliased shapes


def conv1_patterns():
    z = lambda: np.zeros((50, 50))
    p = {}
    p["ridge_h"] = z(); p["ridge_h"][24:26] = -1
    p["ridge_v"] = z(); p["ridge_v"][:, 24:26] = -1
    r, c = np.mgrid[:50, :50]
    p["ridge_d45"] = np.where(r + c == 48, -1.0, 0.0)
    p["ridge_d135"] = np.where(r == c, -1.0, 0.0)
    p["bright_h"] = z(); p["bright_h"][24:26] = 1
    p["dark"] = z(); p["dark"][20:29, 20:29] = -1
    p["bright"] = z(); p["bright"][24, 20:29] = 1
    p["edge_top"] = z(); p["edge_top"][25:] = -1
    p["edge_bottom"] = z(); p["edge_bottom"][:25] = -1
    return p


def draw(fn, contrast=1.5):
    """Render fn(ImageDraw, s) on a 50x50 canvas; dark ink = -contrast, bright = +contrast."""
    img = Image.new("L", (50 * S, 50 * S), 128)
    fn(ImageDraw.Draw(img), S)
    a = np.asarray(img.resize((50, 50), Image.LANCZOS), float)
    return (a - 128) / 127 * contrast


def curve(d, s, x0, x1, y_of_t, width, ink=0):
    xs = np.linspace(x0, x1, 40)
    t = (xs - (x0 + x1) / 2) / ((x1 - x0) / 2)
    d.line([(x * s, y_of_t(tt) * s) for x, tt in zip(xs, t)], fill=ink, width=int(width * s))


def conv2_shapes():
    return {
        "straight lip": lambda d, s: curve(d, s, 13, 37, lambda t: 36, 2),
        "smile": lambda d, s: curve(d, s, 13, 37, lambda t: 36 - 4 * t * t, 2),
        "frown": lambda d, s: curve(d, s, 13, 37, lambda t: 34 + 4 * t * t, 2),
        "open mouth": lambda d, s: d.ellipse([18 * s, 32 * s, 32 * s, 41 * s], fill=0),
        "teeth": lambda d, s: (d.ellipse([15 * s, 32 * s, 35 * s, 40 * s], fill=0),
                               d.rectangle([17 * s, 34 * s, 33 * s, 35.5 * s], fill=255)),
        "furrow": lambda d, s: d.line([(25 * s, 9 * s), (25 * s, 17 * s)], fill=0,
                                      width=int(1.5 * s)),
        "wrinkles": lambda d, s: [d.line([(a * s, b * s), (c * s, e * s)], fill=0, width=s)
                                  for a, b, c, e in [(20, 18, 28, 19), (21, 21, 29, 21),
                                                     (20, 24, 28, 23), (21, 27, 29, 27)]],
        "arch": lambda d, s: curve(d, s, 13, 27, lambda t: 10 + 3 * t * t, 2.5),
    }


def table(rows, cols, values, fmt="{:6.2f}"):
    w = max(len(r) for r in rows)
    print(" " * w + "".join(f"{c[:10]:>11}" for c in cols))
    for r, v in zip(rows, values):
        print(f"{r:<{w}}" + "".join(f"{fmt.format(x):>11}" for x in v))


def main():
    print("conv1 linear response at the pattern centre (rows: pattern, cols: filter)")
    pats = conv1_patterns()
    x = np.stack(list(pats.values()))[:, None]
    lin = M.conv2d(x, M.K1, np.zeros(len(M.K1)))[:, :, 24, 24]
    table(list(pats), M.CONV1_NAMES, lin)

    print("\nconv2 max response after conv1 + ReLU + pool (rows: shape, cols: part)")
    shapes = conv2_shapes()
    x = np.stack([draw(f) for f in shapes.values()])[:, None]
    _, _, c2 = M.parts(x)
    table(list(shapes), M.CONV2_NAMES, c2.max(axis=(2, 3)))


if __name__ == "__main__":
    main()
