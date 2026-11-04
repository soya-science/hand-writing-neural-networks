"""Every weight of the few-shot head on top of the frozen ResNet-50 (zero_shot_resnet/backbone.py).

NOT zero-shot: this head was set from 10 training images per class (shots.txt, picked at
random by select_shots.py). Nothing else from train, validation or test was looked at.

What was decided by hand, after looking at the 70 shots (figures/shots_*.png):
- SHOT_LABELS: which facial actions each shot shows, marked by eye;
- which action units to use, their zones and pooling (zones as in the zero-shot head), the
  16-channel detector size, and the naive-Bayes form of the fc layer.
What was measured on the shots (measure_shots.py, pasted here as literals):
- each detector's channels, resting levels and gains (from my labels, not the emotion labels);
- the fc log-odds, counted from my labels per emotion class.

The head:
  zone pool  v_c = mean or max of layer3 channel c over the unit's zone
  evidence   e = mean over the unit's channels of (v_c - rest_c) / gain_c
             (about 0 on shots without the action, about 1 on shots with it)
  AU         u = clip(e, 0, 1)
  logits     u @ FC + FC_BIAS: naive Bayes with log P(unit | class), uniform class priors
"""

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# --- My labels for the 70 shots, in shots.txt order ----------------------------------------
# R brows raised (inner, outer or both)   L brows lowered or knit   W eyes wide
# S eyes squeezed or narrowed   N nose wrinkle or sneering upper lip
# U lip corners up   D lip corners down   O mouth open   T teeth showing
SHOT_LABELS = {
    "angry":    ["LSNOT", "LSND", "LSD", "LOT", "LSNT", "LSNT", "LOT", "LSNT", "LSNT", "NOT"],
    "disgust":  ["", "LNT", "SD", "LD", "LND", "LS", "LSN", "S", "SD", "SNOT"],
    "fear":     ["RW", "RLWT", "O", "RWO", "RWOT", "SOT", "RWT", "RWO", "RWOT", "RWOT"],
    "happy":    ["U", "US", "UOT", "UOT", "UOT", "USOT", "U", "US", "US", "UOT"],
    "neutral":  ["", "", "", "O", "", "", "", "", "", ""],
    "sad":      ["LSD", "SD", "LSO", "RD", "LSO", "LSOT", "SOT", "LSO", "L", "SOT"],
    "surprise": ["RO", "R", "RWO", "RWO", "RW", "RW", "O", "RWO", "WO", "WO"],
}
AU_CODES = {"brows_raised": "R", "brows_lowered": "L", "eyes_wide": "W", "eyes_squeezed": "S",
            "sneer": "N", "lip_corners_up": "U", "lip_corners_down": "D", "mouth_open": "O",
            "teeth": "T"}

# --- Face zones on the 14x14 layer3 grid (same boxes as the zero-shot head) ----------------
# The shots confirmed the placement: eyes, brows, nose and mouth fall in their boxes on the
# aligned real faces.
ZONES = {
    "forehead":   [(1, 3, 3, 10)],
    "glabella":   [(3, 5, 6, 8)],
    "eyes":       [(4, 5, 2, 4), (4, 5, 9, 11)],
    "eyes_tall":  [(3, 5, 2, 4), (3, 5, 9, 11)],
    "nose":       [(6, 7, 6, 7)],
    "mouth":      [(8, 12, 4, 9)],
    "mouth_mid":  [(9, 11, 3, 10)],
    "mouth_low":  [(10, 12, 3, 10)],
    "teeth":      [(9, 11, 5, 8)],
}

# --- Action units: (zone, pool, channels); channels: (layer3 channel, rest, gain) ----------
# From shots_report.txt, three significant digits. A negative gain means the channel drops
# when the action is present.
AUS = {
    "brows_raised": ("forehead", "mean", [
        (33, 0.631, 0.583), (982, 0.306, -0.242), (126, 0.892, -0.262), (236, 1.81, -0.854),
        (98, 0.562, 0.179), (38, 2.28, 0.599), (111, 0.33, -0.223), (197, 0.0655, -0.0308),
        (35, 1.19, 0.282), (966, 1.24, -0.265), (497, 0.0307, 0.0961), (694, 0.0176, 0.103),
        (463, 0.461, -0.264), (710, 0.773, -0.367), (932, 0.592, -0.219), (48, 0.135, -0.0933),
    ]),
    "brows_lowered": ("glabella", "max", [
        (411, 0.501, -0.294), (350, 0.867, 1.03), (575, 1.87, 1.0), (257, 0.604, -0.516),
        (853, 7.63, -2.78), (800, 0.895, -0.836), (478, 4.75, -0.874), (966, 1.54, 0.833),
        (872, 2.74, 2.23), (941, 1.65, -0.829), (810, 0.764, 1.36), (826, 2.88, 2.35),
        (712, 0.637, 1.57), (629, 0.406, -0.224), (176, 0.215, 0.293), (371, 2.01, -0.892),
    ]),
    "eyes_wide": ("eyes", "mean", [
        (431, 0.534, -0.365), (324, 0.659, 1.61), (195, 0.588, 1.96), (339, 3.88, 2.28),
        (130, 0.641, -0.332), (672, 3.34, 2.5), (103, 0.666, -0.512), (1007, 1.01, -0.479),
        (137, 4.3, -2.06), (984, 2.11, 1.71), (873, 1.26, -0.978), (791, 4.01, 1.2),
        (874, 1.44, 1.89), (817, 1.57, -0.271), (543, 0.682, -0.401), (924, 0.794, 0.76),
    ]),
    "eyes_squeezed": ("eyes_tall", "mean", [
        (874, 2.18, -1.65), (195, 1.33, -1.15), (339, 4.32, -1.55), (672, 3.3, -1.69),
        (213, 0.493, -0.213), (984, 2.22, -1.1), (826, 1.62, 1.62), (924, 1.06, -0.518),
        (56, 3.89, -1.73), (968, 0.987, -0.574), (265, 0.128, 0.302), (1007, 0.889, 0.377),
        (882, 1.4, -0.877), (291, 0.857, -0.655), (295, 0.322, -0.157), (929, 1.75, 0.627),
    ]),
    "sneer": ("nose", "max", [
        (902, 1.27, 0.895), (941, 0.855, -0.706), (808, 0.296, -0.216), (308, 0.225, -0.164),
        (965, 1.37, -0.709), (224, 0.609, 1.12), (697, 4.21, -1.82), (350, 0.733, 0.764),
        (32, 2.21, -1.32), (668, 1.12, -0.774), (647, 1.32, -0.605), (803, 2.72, -0.979),
        (238, 0.496, -0.393), (459, 5.24, -2.84), (894, 0.796, 1.43), (226, 0.785, 0.777),
    ]),
    "lip_corners_up": ("mouth_mid", "max", [
        (596, 3.24, -2.56), (42, 1.63, -1.43), (429, 1.08, -0.854), (876, 3.66, -1.21),
        (237, 2.12, 1.67), (38, 4.34, -0.738), (1015, 2.53, 2.12), (29, 2.08, -1.75),
        (272, 1.06, -0.889), (486, 2.02, 2.75), (1013, 1.37, -0.363), (1022, 2.28, -0.926),
        (859, 1.9, -1.46), (846, 4.67, 1.45), (338, 6.52, -0.349), (787, 1.94, 2.01),
    ]),
    "lip_corners_down": ("mouth_low", "max", [
        (176, 0.356, -0.329), (113, 3.42, -1.83), (685, 2.79, -1.38), (291, 2.71, -2.36),
        (953, 3.3, -1.72), (719, 3.56, -1.5), (195, 1.67, -1.67), (30, 4.86, 2.61),
        (2, 0.92, -0.796), (78, 1.49, -0.969), (230, 0.728, 0.74), (663, 1.41, -1.14),
        (51, 3.32, -0.456), (264, 6.13, -1.66), (530, 2.25, -1.76), (774, 1.45, -1.25),
    ]),
    "mouth_open": ("mouth", "mean", [
        (339, 1.67, 0.709), (205, 1.29, 0.739), (195, 0.0364, 0.255), (513, 0.755, -0.297),
        (291, 0.162, 0.437), (232, 0.364, 0.234), (354, 1.66, -0.633), (926, 1.52, 0.528),
        (619, 0.0986, 0.369), (426, 0.301, 0.596), (668, 0.525, 0.296), (868, 0.191, 0.209),
        (475, 1.57, -0.491), (767, 0.204, 0.259), (26, 1.28, -0.225), (685, 0.22, 0.209),
    ]),
    "teeth": ("teeth", "max", [
        (346, 0.0255, 0.652), (305, 2.04, 2.45), (339, 2.89, 2.49), (336, 2.86, 2.92),
        (636, 0.168, 3.68), (455, 0.214, 2.04), (286, 5.67, -2.07), (953, 2.36, 2.06),
        (232, 1.0, 0.562), (291, 0.986, 2.84), (964, 1.05, 2.21), (663, 0.315, 1.44),
        (935, 0.651, 1.15), (303, 5.36, 4.69), (278, 5.91, -3.18), (780, 1.99, 2.86),
    ]),
}

# --- fc layer: naive-Bayes log-odds from SHOT_LABELS ---------------------------------------
# FC[au][k] = log(P / (1 - P)) and FC_BIAS[k] = sum over units of log(1 - P), with
# P = P(unit | class k) = (count + 0.5) / (10 + 1). An empty count gives -3.04: seeing the unit
# is strong evidence against that class. No class prior (uniform), as in both zero-shot heads.
#                        angry  disgust     fear    happy  neutral      sad  surprise
FC = {
    "brows_raised":      [  -3.04,   -3.04,    1.22,   -3.04,   -3.04,   -1.85,    0.76],
    "brows_lowered":     [   1.85,    0.00,   -1.85,   -3.04,   -3.04,    0.37,   -3.04],
    "eyes_wide":         [  -3.04,   -3.04,    1.22,   -3.04,   -3.04,   -3.04,    0.76],
    "eyes_squeezed":     [   0.76,    0.37,   -1.85,   -0.37,   -3.04,    1.22,   -3.04],
    "sneer":             [   0.76,   -0.37,   -3.04,   -3.04,   -3.04,   -3.04,   -3.04],
    "lip_corners_up":    [  -3.04,   -3.04,   -3.04,    3.04,   -3.04,   -3.04,   -3.04],
    "lip_corners_down":  [  -1.22,   -0.37,   -3.04,   -3.04,   -3.04,   -0.76,   -3.04],
    "mouth_open":        [  -0.37,   -1.85,    0.76,    0.00,   -1.85,    0.37,    0.76],
    "teeth":             [   1.22,   -1.22,    0.37,    0.00,   -3.04,   -0.76,   -3.04],
}
FC_BIAS =             [  -6.69,   -3.18,   -5.43,   -5.24,   -0.52,   -4.32,   -3.71]
