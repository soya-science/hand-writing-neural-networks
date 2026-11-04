"""One-shot evaluation of the hand-wired network on the real test split.

Agreed protocol: this runs once, the numbers are reported as they come out, and no weight
is changed afterwards.
"""
import os
import sys

import numpy as np
from PIL import Image

import model as M
from weights import CLASSES

DEFAULT_SPLIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset", "test")


def load_split(root):
    images, labels = [], []
    for k, name in enumerate(CLASSES):
        folder = os.path.join(root, name)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                im = Image.open(os.path.join(folder, f)).convert("RGB")
                if im.size != (100, 100):
                    im = im.resize((100, 100), Image.BILINEAR)
                images.append(np.asarray(im))
                labels.append(k)
    return np.stack(images), np.array(labels)


def report(y, pred, title):
    """Print accuracy, balanced accuracy, baselines and the confusion matrix."""
    n = len(CLASSES)
    cm = np.zeros((n, n), int)
    np.add.at(cm, (y, pred), 1)
    support = cm.sum(1)
    recall = cm.diagonal() / np.maximum(support, 1)
    precision = cm.diagonal() / np.maximum(cm.sum(0), 1)
    print(title)
    print(f"  images {cm.sum()}   accuracy {cm.trace() / cm.sum():.1%}   "
          f"balanced accuracy {recall.mean():.1%}")
    print(f"  baselines: chance balanced accuracy {1 / n:.1%}; always '{CLASSES[support.argmax()]}' "
          f"accuracy {support.max() / support.sum():.1%}")
    print("\n  rows = true, cols = predicted")
    print("  " + " " * 9 + "".join(f"{c[:7]:>8}" for c in CLASSES) + "   recall  precision")
    for i, c in enumerate(CLASSES):
        print(f"  {c:<9}" + "".join(f"{v:>8}" for v in cm[i])
              + f"   {recall[i]:6.1%}  {precision[i]:8.1%}")
    return cm


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SPLIT
    X, y = load_split(root)
    pred = M.predict(X)
    report(y, pred, f"Hand-wired CNN on {os.path.normpath(root)}")


if __name__ == "__main__":
    main()
