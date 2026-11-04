"""One-shot evaluation of the ResNet-50 + hand-typed head on the real test split.

Agreed protocol (same as v1): this runs once, the numbers are reported as they come out, and
no head weight is changed afterwards. The script writes results_test.txt and refuses to run
again while that file exists.
"""
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np

import head as H

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "zero_shot", "emotion_cnn"))
from evaluate import load_split, report  # noqa: E402  (v1, unchanged)
from weights import CLASSES  # noqa: E402

TEST_SPLIT = os.path.join(HERE, "..", "dataset", "test")
RESULTS = os.path.join(HERE, "results_test.txt")


def main():
    if os.path.exists(RESULTS):
        sys.exit(f"{RESULTS} exists: the test split has already been used once.")
    X, y = load_split(TEST_SPLIT)
    out = [H.forward(X[i:i + 256]) for i in range(0, len(X), 256)]
    au = np.concatenate([o["au"] for o in out])
    pred = np.concatenate([o["logits"] for o in out]).argmax(1)

    buf = io.StringIO()
    with redirect_stdout(buf):
        report(y, pred, f"ResNet-50 + hand-typed head on {os.path.normpath(TEST_SPLIT)}")
        print("\n  mean AU per true class")
        print("  " + " " * 17 + "".join(f"{c[:7]:>8}" for c in CLASSES))
        for a, name in enumerate(H.AU_NAMES):
            print(f"  {name:<17}" + "".join(f"{au[y == k, a].mean():8.2f}"
                                           for k in range(len(CLASSES))))
    text = buf.getvalue()
    print(text, end="")
    with open(RESULTS, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
