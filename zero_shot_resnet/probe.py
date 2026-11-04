"""Probe the frozen backbone with cartoon twins and rank layer3 channels per action unit.

For each action, twins (the same face at rest and with the action on) are drawn in both photo
regimes. Each channel is pooled over the action's zone and scored by:
  d'      separation of active vs resting faces, (mean_on - mean_rest) / pooled std, per regime
  cross   the largest d' the same pooled channel shows for any other action (want low)
  shift   how far its resting level moves between the standard and rough regimes, in stds
  score   min(d'_standard, d'_rough) - max(cross, 0) - shift / 2
and reported with the two numbers the head needs: rest (mean resting level) and gain (mean
change when the action is on). The channels in head_weights.py were typed from this report.

Writes probe_report.txt next to this file. No real image is used.
"""
import os

import numpy as np

import backbone as BB
import cartoon as C
import head_weights as HW

N_TWINS = 128
SEED = 2000
TOP = 8
# Actions that come with another action built in: an open mouth is shown with teeth and a
# raised upper lip also opens the mouth, so a mouth_open channel may answer to them too.
CROSS_IGNORE = {"mouth_open": {"teeth", "upper_lip_raise"},
                "eyes_wide": {"eyes_narrow"}, "eyes_narrow": {"eyes_wide"}}
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_report.txt")


def zone_mask(name):
    m = np.zeros((BB.GRID, BB.GRID), bool)
    for r0, r1, c0, c1 in HW.ZONES[name]:
        m[r0:r1 + 1, c0:c1 + 1] = True
    return m


def pooled_twins():
    """{(regime, action, state, au): (N, 1024)} with state 0 = rest, 1 = action on."""
    pools = {au: (zone_mask(zone), op) for au, (zone, op, _) in HW.AUS.items()}
    out = {}
    for r, regime in enumerate(C.REGIMES):
        for j, action in enumerate(C.ACTIONS):
            pair = C.twins(action, N_TWINS, SEED + 100 * r + j, regime)
            for state, X in enumerate(pair):
                F = BB.layer3(X)
                for au, (m, op) in pools.items():
                    inside = F[:, :, m]
                    out[regime, action, state, au] = inside.max(-1) if op == "max" else inside.mean(-1)
    return out


def dprime(rest, on, floor):
    return (on.mean(0) - rest.mean(0)) / np.sqrt((rest.var(0) + on.var(0)) / 2 + floor ** 2)


def rank(P, au):
    """Per-channel statistics for one action unit, over all 1024 channels."""
    rest = {g: np.concatenate([P[g, a, 0, au] for a in C.ACTIONS]) for g in C.REGIMES}
    floor = 0.1 * rest["standard"].std(0) + 1e-4  # keeps channels silent at rest from exploding
    d = {g: dprime(P[g, au, 0, au], P[g, au, 1, au], floor) for g in C.REGIMES}
    cross = np.max([np.maximum(*(dprime(P[g, b, 0, au], P[g, b, 1, au], floor) for g in C.REGIMES))
                    for b in C.ACTIONS if b != au and b not in CROSS_IGNORE.get(au, ())], axis=0)
    rs, rr = rest["standard"], rest["rough"]
    shift = np.abs(rr.mean(0) - rs.mean(0)) / np.sqrt((rr.var(0) + rs.var(0)) / 2 + floor ** 2)
    score = np.minimum(d["standard"], d["rough"]) - np.maximum(cross, 0) - shift / 2
    level = np.concatenate([rs, rr]).mean(0)
    gain = np.mean([P[g, au, 1, au].mean(0) - P[g, au, 0, au].mean(0) for g in C.REGIMES], axis=0)
    return dict(score=score, d_std=d["standard"], d_rough=d["rough"], cross=cross, shift=shift,
                rest=level, gain=gain)


def main():
    P = pooled_twins()
    lines = [f"Cartoon probe of ResNet-50 layer3: {N_TWINS} twins per action per regime, seed {SEED}",
             f"columns: channel, score, d' standard, d' rough, cross, shift, rest, gain", ""]
    for au, (zone, op, _) in HW.AUS.items():
        s = rank(P, au)
        lines.append(f"{au}  ({op} over zone '{zone}' {HW.ZONES[zone]})")
        lines.append("    ch   score  d'std  d'rough  cross  shift      rest      gain")
        for c in np.argsort(-s["score"])[:TOP]:
            lines.append(f"  {c:4d}  {s['score'][c]:6.2f} {s['d_std'][c]:6.2f}  {s['d_rough'][c]:6.2f}"
                         f"  {s['cross'][c]:5.2f}  {s['shift'][c]:5.2f}  {s['rest'][c]:8.4f}"
                         f"  {s['gain'][c]:8.4f}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(REPORT, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
