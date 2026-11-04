"""Every weight of the classifier head on top of the frozen ResNet-50, typed by hand.

The backbone (backbone.py) is ImageNet ResNet-50 cut after layer3: 1024 channels on a 14x14
grid. Its channels carry no labels, so they were read off by probing it with cartoon twins
(cartoon.py, probe.py): the same drawn face at rest and with one facial action switched on.
probe_report.txt lists, per action, the channels that change most inside the action's zone,
how specific they are and how stable under rougher photos; the channel numbers, resting
levels and gains below are copied from it, rounded. No real image was looked at.

The head:
  zone pool  v_c = mean or max of layer3 channel c over the action's zone
  evidence   e = mean over the action's channels of (v_c - rest_c) / gain_c
             (about 0 on a cartoon face at rest, about 1 with the cartoon action fully on)
  AU         clip((e - AU_THRESHOLD) / (1 - AU_THRESHOLD), 0, 1)
  logits     AU vector @ FC + FC_BIAS
"""

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# --- Face zones on the 14x14 layer3 grid -------------------------------------------------
# Boxes (r0, r1, c0, c1), inclusive; one cell is about 7x7 px of the 100x100 face. Placed on
# the v1 mean-face geometry (brows near rows 3-4, eyes rows 4-5 at cols 3 and 10, nose base
# row 8, lip line row 10), then the best of a few candidate boxes per action on the cartoon
# twins. The forehead box starts at row 1, not 0, to stay clear of hair.
ZONES = {
    "forehead":   [(1, 3, 3, 10)],
    "brows":      [(2, 4, 2, 5), (2, 4, 8, 11)],
    "glabella":   [(3, 5, 6, 8)],                  # between the brows
    "eyes":       [(4, 5, 2, 4), (4, 5, 9, 11)],
    "eyes_tall":  [(3, 5, 2, 4), (3, 5, 9, 11)],   # eyes plus the lid crease above
    "nose":       [(6, 7, 6, 7)],                  # nose bridge
    "upper_lip":  [(9, 10, 5, 8)],
    "mouth":      [(8, 12, 4, 9)],
    "mouth_mid":  [(9, 11, 3, 10)],                # lip line, corner to corner
    "mouth_low":  [(10, 12, 3, 10)],               # lip line and below: drooping corners
    "teeth":      [(9, 11, 5, 8)],
}

# --- Action units: (zone, pool, channels) --------------------------------------------------
# channels: (layer3 channel, rest, gain), from probe_report.txt, rounded to two digits.
# Picking rule: the four best-scored channels, skipping any whose cross-talk is more than half
# its weaker d', whose resting level shifts more than 0.9 std between photo regimes, or whose
# gain is under a quarter of its resting level (a small shift on real faces would swamp it).
AUS = {
    "brow_raise": ("forehead", "mean", [           # AU1+2
        (500, 0.69, 1.3), (273, 1.2, 0.64), (195, 0.78, 0.97), (305, 0.77, 0.77)]),
    "brows_inner_up": ("brows", "max", [           # AU1
        (679, 3.2, 1.2), (862, 3.6, 2.0), (386, 0.84, 1.7), (401, 10.0, 3.1)]),
    "brow_lower": ("glabella", "max", [            # AU4
        (142, 2.9, 5.3), (636, 0.025, 3.9), (647, 2.4, 4.5), (953, 0.51, 3.0)]),
    "eyes_wide": ("eyes", "mean", [                # AU5
        (933, 3.1, 1.8), (575, 0.82, 0.63), (796, 0.39, 0.81), (291, 1.6, 0.83)]),
    "eyes_narrow": ("eyes_tall", "mean", [         # AU6/7; 408 and 766 fail the gain rule
        (613, 0.70, 0.45), (667, 0.16, 0.21), (929, 2.5, 0.76), (842, 0.029, 0.097)]),
    "nose_wrinkle": ("nose", "max", [              # AU9; 30 fails the shift rule
        (953, 0.63, 6.4), (613, 0.91, 3.8), (647, 3.0, 7.5), (336, 0.16, 2.8)]),
    "upper_lip_raise": ("upper_lip", "mean", [     # AU10
        (1008, 1.2, 1.6), (573, 2.4, 1.9), (396, 0.29, 0.56), (174, 0.016, 0.062)]),
    "lip_corners_up": ("mouth_mid", "max", [       # AU12
        (237, 2.9, 4.0), (142, 4.8, 3.7), (892, 0.41, 1.5), (163, 2.8, 2.8)]),
    "lip_corners_down": ("mouth_low", "max", [     # AU15; 672 fails the cross-talk rule
        (328, 2.4, 6.4), (671, 2.4, 5.1), (311, 0.96, 1.5), (915, 0.96, 0.59)]),
    "mouth_open": ("mouth", "mean", [              # AU26/27
        (129, 0.54, 1.5), (398, 0.41, 0.98), (113, 0.49, 1.2), (988, 0.091, 0.18)]),
    "teeth": ("teeth", "max", [                    # AU25 with teeth; 633 shift, 303 cross-talk
        (252, 0.013, 2.2), (601, 1.4, 1.9), (125, 1.2, 1.1), (139, 0.93, 1.4)]),
}

# Evidence below this counts as nothing. On cartoon emotion faces 0.4 was best in the standard
# regime (90% balanced vs 89% at 0.3) and close in the rough one (82% vs 84%); real faces are
# likely noisier still, which argues for the higher cut.
AU_THRESHOLD = 0.4

# --- fc layer, action units -> emotion logits (EMFACS prototypes) ---------------------------
# The same table as v1 (zero_shot/emotion_cnn/weights.py), so the two experiments differ only
# in how the action units are detected. Neutral loses half a unit per unit of any AU.
#                     angry disgust fear happy neutral sad surprise
FC = {
    "brow_raise":       [-1,   -1,    2,    0,   -0.5,  0,    3],  # surprise; fear too
    "brows_inner_up":   [-1,    0,    2,    0,   -0.5,  3,    0],  # the sad brow; fear too
    "brow_lower":       [ 3,    1,    1,   -1,   -0.5,  0,   -2],  # anger's main sign
    "eyes_wide":        [ 0,   -1,    2,   -1,   -0.5, -1,    1],  # fear and surprise
    "eyes_narrow":      [ 1,    1,   -1,    1,   -0.5,  1,   -1],  # glare, sneer, smiling cheeks
    "nose_wrinkle":     [ 0,    3,    0,    0,   -0.5,  0,    0],  # disgust's main sign
    "upper_lip_raise":  [ 0,    2,    0,    1,   -0.5,  0,    0],  # disgust; open smiles too
    "lip_corners_up":   [-1,   -1,    0,    3,   -0.5, -2,    0],  # happy's main sign
    "lip_corners_down": [ 0,    1,    0,   -2,   -0.5,  3,   -1],  # sad; disgust too
    "teeth":            [ 1,    1,    1,    2,   -0.5, -1,   -1],  # open smile, snarl, fear grimace
    "mouth_open":       [ 0,    0,    1,    0,   -0.5, -1,    3],  # surprise's jaw drop
}
#                     angry disgust fear happy neutral sad surprise
FC_BIAS =            [ 0,    0,    0,    0,    0.8,  0,    0]
