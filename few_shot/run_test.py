"""One-shot evaluation of the few-shot head on the real test split.

Same protocol as the zero-shot experiments: this runs once, the numbers are reported as they
come out, and no weight is changed afterwards. The script writes results_test.txt and refuses
to run again while that file exists.

Alongside the hand-assigned head it reports a reference fixed before the run: weight
imprinting, the standard few-shot recipe. Each class's weight vector is the mean ResNet-50
feature (layer4, global average pool) of its 10 shots, centred on the mean of all 70 and
L2-normalised; a face goes to the class with the highest cosine. Those weights are computed,
not typed; the reference shows what the same 70 images give without any hand-made choices.
"""
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import torch

import few_shot_head as H
import select_shots as SS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "zero_shot", "emotion_cnn"))
from evaluate import load_split, report  # noqa: E402  (v1, unchanged)
from weights import CLASSES  # noqa: E402

TEST_SPLIT = os.path.join(HERE, "..", "dataset", "test")
RESULTS = os.path.join(HERE, "results_test.txt")


@torch.no_grad()
def global_features(rgb, batch=128):
    net = H.BB._network()
    out = []
    for i in range(0, len(rgb), batch):
        maps = torch.from_numpy(H.BB.layer3(rgb[i:i + batch])).to(H.BB.DEVICE)
        out.append(net.layer4(maps).mean((2, 3)).cpu().numpy())
    return np.concatenate(out)


def imprinted(shots_X, shots_y):
    f = global_features(shots_X)
    centre = f.mean(0)
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    W = unit(np.stack([unit(f[shots_y == k] - centre).mean(0) for k in range(len(CLASSES))]))
    return lambda X: (unit(global_features(X) - centre) @ W.T).argmax(1)


def main():
    if os.path.exists(RESULTS):
        sys.exit(f"{RESULTS} exists: the test split has already been used once.")
    shots_X, shots_y = SS.load(SS.read_shots())
    baseline = imprinted(shots_X, shots_y)
    X, y = load_split(TEST_SPLIT)
    out = [H.forward(X[i:i + 256]) for i in range(0, len(X), 256)]
    au = np.concatenate([o["au"] for o in out])
    pred = np.concatenate([o["logits"] for o in out]).argmax(1)

    buf = io.StringIO()
    with redirect_stdout(buf):
        report(y, pred, f"Few-shot head (10 shots per class) on {os.path.normpath(TEST_SPLIT)}")
        print("\n  mean AU per true class")
        print("  " + " " * 17 + "".join(f"{c[:7]:>8}" for c in CLASSES))
        for a, name in enumerate(H.AU_NAMES):
            print(f"  {name:<17}" + "".join(f"{au[y == k, a].mean():8.2f}"
                                           for k in range(len(CLASSES))))
        print()
        report(y, baseline(X), "Reference: weight imprinting from the same 70 shots")
    text = buf.getvalue()
    print(text, end="")
    with open(RESULTS, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
