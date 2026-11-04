"""Save pictures of every layer's activations for one synthetic face per emotion.

figures/feature_maps.png: one row per emotion; columns are the input, the nine conv1 maps
and the ten conv2 part maps (brighter = stronger, all on the same 0..1 scale).
figures/zones.png: the face zones drawn over a synthetic neutral face.
"""
import os

import numpy as np
from PIL import Image, ImageDraw

import model as M
import synthetic_faces as SF
import weights as W

TILE = 100
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def _tile(a, lo=0.0, hi=1.0):
    a = np.clip((a - lo) / (hi - lo), 0, 1) * 255
    return Image.fromarray(a.astype(np.uint8)).resize((TILE, TILE), Image.NEAREST)


def feature_sheet(seed=4):
    rng = np.random.default_rng(seed)
    X = np.stack([SF.make_face(c, rng) for c in W.CLASSES])
    out = M.forward(X)
    names = ["input"] + M.CONV1_NAMES + M.CONV2_NAMES
    sheet = Image.new("L", (TILE * len(names) + 70, TILE * len(W.CLASSES) + 16), 255)
    d = ImageDraw.Draw(sheet)
    for j, n in enumerate(names):
        d.text((70 + TILE * j + 2, 2), n[:16], fill=0)
    for i, c in enumerate(W.CLASSES):
        y = 16 + TILE * i
        pred = W.CLASSES[out["logits"][i].argmax()]
        d.text((2, y + 40), c, fill=0)
        d.text((2, y + 54), f"-> {pred}", fill=0)
        sheet.paste(Image.fromarray(X[i]).convert("L"), (70, y))
        maps = list(out["conv1"][i]) + list(out["conv2"][i])
        for j, m in enumerate(maps, start=1):
            sheet.paste(_tile(m), (70 + TILE * j, y))
    return sheet


def zone_sheet(seed=4):
    face = SF.make_face("neutral", np.random.default_rng(seed))
    img = Image.fromarray(face).resize((400, 400), Image.BICUBIC)
    d = ImageDraw.Draw(img)
    colors = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
              "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#800000",
              "#aaffc3", "#808000"]
    for k, (name, boxes) in enumerate(W.ZONES.items()):
        col = colors[k % len(colors)]
        for r0, r1, c0, c1 in boxes:  # 25x25 cell -> 16 px at 400x400
            d.rectangle([c0 * 16, r0 * 16, (c1 + 1) * 16 - 1, (r1 + 1) * 16 - 1], outline=col)
        r0, _, c0, _ = boxes[0]
        d.text((c0 * 16 + 2, r0 * 16 + 1), name, fill=col)
    return img


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    for name, img in [("feature_maps.png", feature_sheet()), ("zones.png", zone_sheet())]:
        path = os.path.join(FIG_DIR, name)
        img.save(path)
        print(path)


if __name__ == "__main__":
    main()
