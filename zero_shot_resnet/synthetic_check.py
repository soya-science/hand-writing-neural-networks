"""Wiring check on fresh cartoon faces (seeds unused by probe.py), in both photo regimes.

1. Twins: mean AU evidence when each action is switched on minus at rest. The diagonal should
   be near 1 and the rest near 0.
2. v1's emotion prototypes: confusion matrix and mean AU per true class.

Like v1's synthetic check, this tests the wiring, not real-world accuracy: the same person
drew the faces and picked the channels.
"""
import os
import sys

import numpy as np

import cartoon as C
import head as H

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zero_shot",
                             "emotion_cnn"))
from evaluate import report  # noqa: E402  (v1, unchanged)
from weights import CLASSES  # noqa: E402

SEED = 9000


def twin_table(regime, n=64):
    print(f"\n  AU evidence, action on minus rest ({regime}); rows: action, cols: AU")
    print("  " + " " * 17 + "".join(f"{a[:7]:>8}" for a in H.AU_NAMES))
    for j, action in enumerate(C.ACTIONS):
        rest, on = C.twins(action, n, SEED + 50 * (regime == "rough") + j, regime)
        d = H.forward(on)["evidence"].mean(0) - H.forward(rest)["evidence"].mean(0)
        print(f"  {action:<17}" + "".join(f"{v:8.2f}" for v in d))


def emotion_check(regime, per_class=60):
    X, y = C.emotion_set(per_class, SEED + 500 + (regime == "rough"), regime)
    out = H.forward(X)
    report(y, out["logits"].argmax(1), f"\nHead on cartoon emotion faces ({regime})")
    print("\n  mean AU per true class")
    print("  " + " " * 17 + "".join(f"{c[:7]:>8}" for c in CLASSES))
    for a, name in enumerate(H.AU_NAMES):
        print(f"  {name:<17}" + "".join(f"{out['au'][y == k, a].mean():8.2f}"
                                       for k in range(len(CLASSES))))


def main():
    for regime in C.REGIMES:
        twin_table(regime)
        emotion_check(regime)


if __name__ == "__main__":
    main()
