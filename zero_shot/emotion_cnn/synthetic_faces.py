"""Schematic 100x100 faces for each emotion, drawn from scratch on the mean-face geometry.

These are the only faces the network is debugged on. They check the wiring (does each
detector fire where it should?), not real-world accuracy: the same person designed both
the faces and the detectors.

Run it to classify a fresh synthetic set and save a sample sheet to figures/.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from weights import CLASSES

S = 4                       # supersampling factor while drawing
CX = 48                     # face centre column (mean face: eyes near cols 23 and 73)
BROW_Y, EYE_Y, MOUTH_Y = 27, 33, 73
EYE_DX = 25
SKIN, BG = 175, 105

NEUTRAL = dict(
    brow_raise=0.0,    # px the whole brow moves up
    brow_tilt=0.0,     # px the inner end moves up (+: tent / \, -: V \ /)
    brow_arch=2.0,     # px the middle of the brow sits above the line through its ends
    furrow=0.0,        # strength of the creases between the brows
    forehead=0.0,      # strength of the forehead creases
    eye_open=7.0,      # eye opening height in px
    nose_wrinkle=0.0,  # strength of the creases across the nose bridge
    upper_lip=0.0,     # px the upper lip lifts in the middle
    curve=0.0,         # px the lip corners sit above the mouth centre (negative: below)
    open=0.0,          # px the mouth opens in the middle
    width=22.0,        # half-width of the mouth in px
    teeth=0.0,         # 1 = upper teeth visible in the opening
    lips=1.0,          # lip thickness factor
    folds=0.4,         # strength of the nose-to-mouth folds
)

PROTOTYPES = {
    "angry":    dict(brow_raise=-3, brow_tilt=-5, furrow=1, eye_open=4.5, lips=0.4, width=20),
    "disgust":  dict(brow_raise=-2, brow_tilt=-2, eye_open=4.5, nose_wrinkle=1, upper_lip=4,
                     curve=-3, open=3, teeth=1, folds=0.8),
    "fear":     dict(brow_raise=4, brow_tilt=4, furrow=0.6, eye_open=11, curve=-1, open=5,
                     width=27, teeth=1),
    "happy":    dict(eye_open=4.5, curve=6, open=7, width=26, teeth=1, folds=1.0),
    "neutral":  dict(),
    "sad":      dict(brow_tilt=5, eye_open=5.5, curve=-5, width=20),
    "surprise": dict(brow_raise=6, brow_arch=5, forehead=1, eye_open=11, open=16, width=14),
}

# Common alternatives, each picked with the given probability.
VARIANTS = {
    "happy": (0.4, dict(open=0, teeth=0)),              # closed-mouth smile
    "angry": (0.3, dict(open=6, teeth=1, lips=0.8)),    # snarl with bared teeth
}

# Person-to-person variation (px) added to the geometric parameters.
SPREAD = dict(brow_raise=0.8, brow_tilt=0.8, brow_arch=0.6, eye_open=0.8, curve=0.8)


def sample_params(emotion, rng):
    p = dict(NEUTRAL, **PROTOTYPES[emotion])
    if emotion in VARIANTS and rng.random() < VARIANTS[emotion][0]:
        p.update(VARIANTS[emotion][1])
    for k in p:
        if k != "teeth":
            p[k] *= rng.uniform(0.75, 1.25)
        p[k] += rng.normal(0, SPREAD.get(k, 0))
    p["eye_open"] = max(p["eye_open"], 2.5)
    p["open"], p["upper_lip"] = max(p["open"], 0), max(p["upper_lip"], 0)
    return p


def _pts(xs, ys):
    return [(x * S, y * S) for x, y in zip(xs, ys)]


def _line(d, xs, ys, ink, width):
    d.line(_pts(xs, ys), fill=int(ink), width=max(1, round(width * S)), joint="curve")


def _mouth_curves(p, t):
    """Lip line, top and bottom of the opening at positions t in [-1, 1] across the mouth."""
    bulge = 1 - t ** 2
    mid = MOUTH_Y - p["curve"] * t ** 2
    return mid, mid - p["upper_lip"] * bulge, mid + p["open"] * bulge ** 0.7


def draw(p):
    """Draw one face at 4x resolution; returns a grayscale PIL image."""
    img = Image.new("L", (100 * S, 100 * S), BG)
    d = ImageDraw.Draw(img)
    d.ellipse([(CX - 37) * S, -22 * S, (CX + 37) * S, 97 * S], fill=SKIN)

    if p["forehead"] > 0:
        for k in range(3):
            xs = np.linspace(CX - 17, CX + 17, 20)
            ys = 9 + 4.5 * k - p["brow_raise"] / 3 + 1.5 * ((xs - CX) / 17) ** 2
            _line(d, xs, ys, SKIN - 50 * p["forehead"], 1.5)

    brow_y = BROW_Y - p["brow_raise"]
    for side in (-1, 1):
        xs = np.linspace(CX + side * 11, CX + side * 37, 24)       # inner end -> outer end
        u = (np.abs(xs - CX) - 11) / 26
        ys = brow_y - p["brow_tilt"] * (1 - u) - p["brow_arch"] * 4 * u * (1 - u)
        _line(d, xs, ys, 70, 4.5)
    if p["furrow"] > 0:
        for side in (-1, 1):
            _line(d, [CX + side * 3.5] * 2, [brow_y - 2, brow_y + 7], SKIN - 70 * p["furrow"], 2)

    for side in (-1, 1):
        ex, h = CX + side * EYE_DX, p["eye_open"]
        box = [(ex - 10) * S, (EYE_Y - h / 2) * S, (ex + 10) * S, (EYE_Y + h / 2) * S]
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).ellipse(box, fill=255)
        eye = Image.new("L", img.size, 225)                         # sclera
        ImageDraw.Draw(eye).ellipse([(ex - 4.5) * S, (EYE_Y - 4.5) * S,
                                     (ex + 4.5) * S, (EYE_Y + 4.5) * S], fill=40)  # iris
        img.paste(eye, mask=mask)
        d.arc(box, 180, 360, fill=35, width=round(1.5 * S))         # upper lid and lashes
        d.arc(box, 0, 180, fill=120, width=S)                       # lower lid

    for side in (-1, 1):
        _line(d, [CX + side * 6, CX + side * 8], [40, 55], SKIN - 25, 2)        # nose sides
        d.ellipse([(CX + side * 6 - 3) * S, 56 * S, (CX + side * 6 + 3) * S, 59 * S], fill=70)
        _line(d, [CX + side * 9, CX + side * (p["width"] + 3)],                 # folds
              [58, MOUTH_Y + 3 - p["curve"] / 2], SKIN - 60 * p["folds"], 1.5)
    if p["nose_wrinkle"] > 0:
        for y in (39, 45, 51):
            xs = np.linspace(CX - 6, CX + 6, 10)
            _line(d, xs, y + ((xs - CX) / 6) ** 2, SKIN - 70 * p["nose_wrinkle"], 2)

    t = np.linspace(-1, 1, 41)
    xs = CX + p["width"] * t
    mid, top, bot = _mouth_curves(p, t)
    lip = np.sqrt(1 - t ** 2) * p["lips"]
    d.polygon(_pts(xs, top - 3 * lip) + _pts(xs[::-1], top[::-1]), fill=SKIN - 45)
    d.polygon(_pts(xs, bot) + _pts(xs[::-1], (bot + 4 * lip)[::-1]), fill=SKIN - 35)
    if (bot - top).max() > 1.5:
        d.polygon(_pts(xs, top) + _pts(xs[::-1], bot[::-1]), fill=35)       # opening
        if p["teeth"] > 0.5:
            tt = t * 0.85
            _, ttop, tbot = _mouth_curves(p, tt)
            txs = CX + p["width"] * tt
            tlow = ttop + np.minimum(5, (tbot - ttop) * 0.45)
            d.polygon(_pts(txs, ttop) + _pts(txs[::-1], tlow[::-1]), fill=225)
    else:
        _line(d, xs, mid, 55, 2 * np.sqrt(p["lips"]))                       # closed lip line
    return img


def _jitter_pose(img, rng):
    """Small random rotation, scale and shift, as left over after face alignment."""
    ang, sc = np.deg2rad(rng.uniform(-3, 3)), rng.uniform(0.95, 1.05)
    tx, ty = rng.uniform(-2, 2, 2) * S
    c = 50 * S
    m00, m01 = np.cos(ang) / sc, np.sin(ang) / sc
    m10, m11 = -m01, m00
    coeffs = (m00, m01, c - m00 * (c + tx) - m01 * (c + ty),
              m10, m11, c - m10 * (c + tx) - m11 * (c + ty))
    return img.transform(img.size, Image.AFFINE, coeffs, Image.BICUBIC, fillcolor=BG)


def _photograph(img, rng):
    """Downsample and add uneven lighting, contrast change, blur and sensor noise."""
    a = np.asarray(img.resize((100, 100), Image.LANCZOS), float)
    yy, xx = np.mgrid[:100, :100] / 99 - 0.5
    a = a + rng.uniform(-25, 25) * xx + rng.uniform(-15, 15) * yy
    a = (a - a.mean()) * rng.uniform(0.8, 1.2) + a.mean() + rng.uniform(-15, 15)
    blurred = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    a = np.asarray(blurred.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0))), float)
    a = np.clip(a + rng.normal(0, 4, a.shape), 0, 255)
    return np.clip(np.stack([a, a * 0.92, a * 0.85], -1), 0, 255).astype(np.uint8)


def make_face(emotion, rng):
    return _photograph(_jitter_pose(draw(sample_params(emotion, rng)), rng), rng)


def make_set(per_class, seed=0):
    rng = np.random.default_rng(seed)
    X = [make_face(c, rng) for c in CLASSES for _ in range(per_class)]
    return np.stack(X), np.repeat(np.arange(len(CLASSES)), per_class)


def main():
    import model as M
    from evaluate import report

    X, y = make_set(per_class=60, seed=1)
    out = M.forward(X)
    report(y, out["logits"].argmax(1), "Hand-wired CNN on synthetic cartoon faces")

    print("\n  mean action-unit evidence per true class")
    print("  " + " " * 17 + "".join(f"{c[:7]:>8}" for c in CLASSES))
    for a, name in enumerate(M.AU_NAMES):
        means = [out["au"][y == k, a].mean() for k in range(len(CLASSES))]
        print(f"  {name:<17}" + "".join(f"{v:8.2f}" for v in means))

    os.makedirs(os.path.join(os.path.dirname(__file__), "figures"), exist_ok=True)
    sheet = Image.new("RGB", (100 * len(CLASSES), 200))
    for k in range(len(CLASSES)):
        for j in range(2):
            sheet.paste(Image.fromarray(X[k * 60 + j]), (100 * k, 100 * j))
    path = os.path.join(os.path.dirname(__file__), "figures", "synthetic_faces.png")
    sheet.resize((sheet.width * 2, sheet.height * 2), Image.NEAREST).save(path)
    print(f"\n  sample sheet: {path}  (columns: {', '.join(CLASSES)})")


if __name__ == "__main__":
    main()
