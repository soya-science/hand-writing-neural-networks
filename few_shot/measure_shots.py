"""Measure the few-shot head's numbers on the 70 shots and print them as Python literals.

For each action unit, the shots I marked as showing it (SHOT_LABELS in few_shot_weights.py)
are compared with the shots I marked as not showing it, channel by channel, after pooling
layer3 over the unit's zone. The 16 channels that separate the two groups best (largest |d'|)
become the unit's detector, with rest = mean over the negatives and gain = mean over the
positives minus rest (negative when the channel drops). The fc table is the naive-Bayes
log-odds of each unit given each class, counted from SHOT_LABELS with half-count smoothing.

The printed AUS and FC blocks were pasted into few_shot_weights.py; this script checks that the
pasted copy still matches. It also prints a leave-one-out estimate over the shots, the only
accuracy figure looked at before the single test run. Writes shots_report.txt.
"""
import os
import sys

import numpy as np

import few_shot_weights as FW
import select_shots as SS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "zero_shot_resnet"))
import backbone as BB  # noqa: E402  (frozen ResNet-50, from the zero-shot experiment)

N_CHANNELS = 16
SMOOTH = 0.5
REPORT = os.path.join(HERE, "shots_report.txt")


def zone_mask(name):
    m = np.zeros((BB.GRID, BB.GRID), bool)
    for r0, r1, c0, c1 in FW.ZONES[name]:
        m[r0:r1 + 1, c0:c1 + 1] = True
    return m


def labels_matrix():
    """(70, n_aus) bool: my labels, in shots.txt order."""
    rows = [s for c in FW.CLASSES for s in FW.SHOT_LABELS[c]]
    return np.array([[FW.AU_CODES[au] in s for au in FW.AUS] for s in rows])


def pooled(maps):
    out = {}
    for au, (zone, op, _) in FW.AUS.items():
        inside = maps[:, :, zone_mask(zone)]
        out[au] = inside.max(-1) if op == "max" else inside.mean(-1)
    return out


def dprime(neg, pos):
    return (pos.mean(0) - neg.mean(0)) / np.sqrt((neg.var(0) + pos.var(0)) / 2 + 1e-6)


def detector(v, lab):
    """v: (n, 1024) pooled channel values, lab: (n,) bool -> channels, rest, gain."""
    ch = np.argsort(-np.abs(dprime(v[~lab], v[lab])))[:N_CHANNELS]
    rest = v[~lab][:, ch].mean(0)
    return ch, rest, v[lab][:, ch].mean(0) - rest


def fc_table(A, y):
    """Naive-Bayes log-odds: logits = sum_au u log P + (1 - u) log(1 - P) = u @ FC + FC_BIAS."""
    P = np.stack([(A[y == k].sum(0) + SMOOTH) / ((y == k).sum() + 2 * SMOOTH)
                  for k in range(len(FW.CLASSES))], 1)  # (n_aus, 7)
    return np.log(P / (1 - P)), np.log(1 - P).sum(0)


def evidence(v_row, ch, rest, gain):
    return ((v_row[ch] - rest) / gain).mean()


def leave_one_out(V, A, y):
    pred = []
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        u = np.array([np.clip(evidence(V[au][i], *detector(V[au][tr], A[tr, a])), 0, 1)
                      for a, au in enumerate(FW.AUS)])
        fc, bias = fc_table(A[tr], y[tr])
        pred.append((u @ fc + bias).argmax())
    return np.array(pred)


def sig3(x):
    return float(f"{x:.3g}")


def main():
    X, y = SS.load(SS.read_shots())
    A = labels_matrix()
    V = pooled(BB.layer3(X))
    lines = ["Few-shot measurements on the 70 shots", ""]

    lines.append("label counts per class (rows: AU, cols: " + ", ".join(FW.CLASSES) + ")")
    for a, au in enumerate(FW.AUS):
        lines.append(f"  {au:<17}" + "".join(f"{A[y == k, a].sum():4d}" for k in range(7)))

    lines += ["", "AUS block (paste into few_shot_weights.py):", "AUS = {"]
    measured = {}
    for a, (au, (zone, op, _)) in enumerate(FW.AUS.items()):
        ch, rest, gain = detector(V[au], A[:, a])
        measured[au] = [(int(c), sig3(r), sig3(g)) for c, r, g in zip(ch, rest, gain)]
        lines.append(f'    "{au}": ("{zone}", "{op}", [')
        for j in range(0, N_CHANNELS, 4):
            lines.append("        " + " ".join(f"({c}, {r}, {g})," for c, r, g in measured[au][j:j + 4]))
        lines.append("    ]),")
    lines.append("}")

    fc, bias = fc_table(A, y)
    lines += ["", "FC block:", "#" + " " * 21 + "  ".join(f"{c[:7]:>7}" for c in FW.CLASSES), "FC = {"]
    for a, au in enumerate(FW.AUS):
        lines.append(f'    "{au}":' + " " * (18 - len(au)) + "[" + ", ".join(f"{v:7.2f}" for v in fc[a]) + "],")
    lines += ["}", "FC_BIAS =" + " " * 13 + "[" + ", ".join(f"{v:7.2f}" for v in bias) + "]"]

    pred = leave_one_out(V, A, y)
    recall = [np.mean(pred[y == k] == k) for k in range(7)]
    lines += ["", f"leave-one-out over the shots: accuracy {np.mean(pred == y):.1%}, per-class recall "
              + ", ".join(f"{c} {r:.0%}" for c, r in zip(FW.CLASSES, recall))]

    same = all(measured[au] == list(chs) for au, (_, _, chs) in FW.AUS.items()) and \
        np.allclose(np.round(fc, 2), [FW.FC[au] for au in FW.AUS]) and \
        np.allclose(np.round(bias, 2), FW.FC_BIAS)
    lines.append(f"few_shot_weights.py matches these measurements: {same}")
    text = "\n".join(lines)
    print(text)
    with open(REPORT, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
