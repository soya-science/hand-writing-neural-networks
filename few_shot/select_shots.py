"""Pick the few-shot examples: 10 training images per class, chosen at random with a fixed seed.

Writes shots.txt (one "class/filename" per line) and, for looking at them, one contact sheet
per class in figures/ with the head's face zones drawn over every face. These 70 images are
the only training images the few-shot head is allowed to see.
"""
import os

import numpy as np
from PIL import Image, ImageDraw

import few_shot_weights as FW

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "..", "dataset", "train")
SHOTS = os.path.join(HERE, "shots.txt")
FIG_DIR = os.path.join(HERE, "figures")
PER_CLASS = 10
SEED = 0
SCALE = 3                     # contact-sheet magnification
CELL = 100 / 14 * SCALE       # one layer3 grid cell in sheet pixels


def pick():
    rng = np.random.default_rng(SEED)
    shots = []
    for c in FW.CLASSES:
        files = sorted(f for f in os.listdir(os.path.join(TRAIN, c))
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        shots += [f"{c}/{files[i]}" for i in sorted(rng.choice(len(files), PER_CLASS, replace=False))]
    return shots


def load(shots):
    X, y = [], []
    for s in shots:
        im = Image.open(os.path.join(TRAIN, s)).convert("RGB")
        X.append(np.asarray(im.resize((100, 100), Image.BILINEAR) if im.size != (100, 100) else im))
        y.append(FW.CLASSES.index(s.split("/")[0]))
    return np.stack(X), np.array(y)


def read_shots():
    with open(SHOTS) as f:
        return [line.strip() for line in f if line.strip()]


def sheet(images, zones=True):
    tile = 100 * SCALE
    out = Image.new("RGB", (tile * 5, tile * ((len(images) + 4) // 5)), "white")
    colors = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
              "#f032e6", "#bcf60c", "#fabebe", "#008080"]
    for k, a in enumerate(images):
        im = Image.fromarray(a).resize((tile, tile), Image.BICUBIC)
        if zones:
            d = ImageDraw.Draw(im)
            for z, (name, boxes) in enumerate(FW.ZONES.items()):
                for r0, r1, c0, c1 in boxes:
                    d.rectangle([c0 * CELL, r0 * CELL, (c1 + 1) * CELL - 1, (r1 + 1) * CELL - 1],
                                outline=colors[z % len(colors)])
        out.paste(im, (tile * (k % 5), tile * (k // 5)))
    return out


def main():
    shots = pick()
    with open(SHOTS, "w") as f:
        f.write("\n".join(shots) + "\n")
    X, y = load(shots)
    os.makedirs(FIG_DIR, exist_ok=True)
    for k, c in enumerate(FW.CLASSES):
        sheet(X[y == k], zones=False).save(os.path.join(FIG_DIR, f"shots_{c}.png"))
        sheet(X[y == k]).save(os.path.join(FIG_DIR, f"shots_{c}_zones.png"))
    print(f"{len(shots)} shots written to {SHOTS}")


if __name__ == "__main__":
    main()
