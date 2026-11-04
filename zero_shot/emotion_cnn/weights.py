"""Every weight of the hand-wired emotion CNN, typed by hand. Nothing here was learned.

Kernels are applied as cross-correlation (no flip), so each grid reads as a picture of the
pattern it looks for: positive cells want brightness, negative cells want darkness.

The only look at the data was a label-free mean of all training faces, used for geometry.
At 50x50 it puts the brows on rows 12-14, the eyes on rows 15-18 (centres near cols 11 and
37), the nose base on rows 27-30 and the lip line on rows 35-37.
"""

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# --- Stage 0: front end ------------------------------------------------------------------
# 1x1 conv, RGB -> gray (ITU-R 601 luma), then a 2x2 average pool to 50x50.
LUMA = [0.299, 0.587, 0.114]

# The 50x50 face is standardized with the mean and std of this box (r0, r1, c0, c1,
# inclusive), which covers brows to chin and cheek to cheek. "Dark" then means darker than
# the skin, not darker than the background.
STANDARDIZE_BOX = (5, 45, 8, 41)

# --- Stage 1: conv1, nine 5x5 filters on the 50x50 face, 'same' padding -----------------
# Each entry: (scale, grid, bias). Every filter except the two brightness-level ones
# ("dark", "bright") sums to zero and outputs 1.0
# on its ideal pattern at a contrast of one standard deviation. Biases ignore faint texture.
# Activation (conv1 and conv2): ReLU capped at 1, so 1.0 means "fully present".
CONV1 = {
    # dark horizontal line 1-3 px thick: brows, eye slits, the lip line, forehead creases
    "ridge_h": (1 / 10, [
        [ 1,  1,  1,  1,  1],
        [ 0,  0,  0,  0,  0],
        [-2, -2, -2, -2, -2],
        [ 0,  0,  0,  0,  0],
        [ 1,  1,  1,  1,  1],
    ], -0.2),
    # dark vertical line: creases between the brows
    "ridge_v": (1 / 10, [
        [ 1,  0, -2,  0,  1],
        [ 1,  0, -2,  0,  1],
        [ 1,  0, -2,  0,  1],
        [ 1,  0, -2,  0,  1],
        [ 1,  0, -2,  0,  1],
    ], -0.2),
    # dark line rising to the right  /
    "ridge_d45": (1 / 10, [
        [ 0,  1,  1,  0, -2],
        [ 1,  1,  0, -2,  0],
        [ 1,  0, -2,  0,  1],
        [ 0, -2,  0,  1,  1],
        [-2,  0,  1,  1,  0],
    ], -0.2),
    # dark line falling to the right  \
    "ridge_d135": (1 / 10, [
        [-2,  0,  1,  1,  0],
        [ 0, -2,  0,  1,  1],
        [ 1,  0, -2,  0,  1],
        [ 1,  1,  0, -2,  0],
        [ 0,  1,  1,  0, -2],
    ], -0.2),
    # bright horizontal band between darker ones: a row of teeth
    "bright_h": (1 / 10, [
        [-1, -1, -1, -1, -1],
        [ 0,  0,  0,  0,  0],
        [ 2,  2,  2,  2,  2],
        [ 0,  0,  0,  0,  0],
        [-1, -1, -1, -1, -1],
    ], -0.2),
    # how much darker than the skin this patch is: pupils, an open mouth. Not zero-sum on
    # purpose; the bias keeps only clearly dark patches (more than half a std below skin).
    "dark": (1 / 25, [
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
    ], -0.5),
    # how bright this short horizontal patch is: teeth, the whites of the eyes. Skin sits
    # near +0.7 std on the standardized face and teeth near +1.5 to +2, so the bias keeps
    # only patches clearly brighter than skin. One row tall, because a band of teeth is
    # only about two pixels tall at this scale.
    "bright": (1 / 3, [
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  1,  1,  1,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
    ], -1.1),
    # bright above, dark below: the top boundary of a dark shape (upper lid, top of an open
    # mouth, upper lip pulled up)
    "edge_top": (1 / 10, [
        [ 1,  1,  1,  1,  1],
        [ 1,  1,  1,  1,  1],
        [ 0,  0,  0,  0,  0],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
    ], -0.2),
    # dark above, bright below: the bottom boundary of a dark shape (lower lid, bottom of an
    # open mouth)
    "edge_bottom": (1 / 10, [
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [ 0,  0,  0,  0,  0],
        [ 1,  1,  1,  1,  1],
        [ 1,  1,  1,  1,  1],
    ], -0.2),
}

# --- Stage 2: conv2, ten 5x5 part detectors on the 25x25 max-pooled conv1 maps -----------
# Each entry: (scale, {conv1 map: grid}, bias). Maps not listed get an all-zero grid.
# Scales make the ideal part output 1.0 before the bias; the bias is the AND threshold.
CONV2 = {
    # a dark hole closed above and below: an open mouth
    "cavity": (1 / 15, {
        "dark": [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        "edge_top": [
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "edge_bottom": [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
        ],
    }, -0.3),
    # a horizontal run of clearly-brighter-than-skin patches: a row of teeth
    "teeth": (1 / 5, {
        "bright": [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    }, -0.2),
    # a long thin dark line: an eye squeezed to a slit
    "slit": (1 / 5, {
        "ridge_h": [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    }, -0.3),
    # a short vertical crease that is not part of a horizontal line
    "furrow": (1 / 3, {
        "ridge_v": [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "ridge_h": [
            [0, 0,  0, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0,  0, 0, 0],
        ],
    }, -0.3),
    # horizontal crease energy: a wrinkled nose, a creased forehead. Close creases weaken
    # each other's ridge responses, so the bright gaps between them (bright_h) count too.
    # Horizontal only, so the near-vertical sides of the nose do not register. It cannot
    # tell one strong line from several faint ones; it is only read on the nose bridge and
    # forehead, which carry no lines on a resting face.
    "wrinkles": (1 / 30, {
        "ridge_h": [
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
        ],
        "bright_h": [
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
        ],
    }, -0.3),
    # left end of a smile: the lip line comes in flat from the right and curls up-left
    #   \ . . . .
    #   . \ . . .
    #   . . - - -
    "smile_end_l": (1 / 3, {
        "ridge_d135": [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "ridge_h": [
            [0, 0, 0,     0,     0],
            [0, 0, 0,     0,     0],
            [0, 0, 0,     0,     0],
            [0, 0, 1 / 3, 1 / 3, 1 / 3],
            [0, 0, 0,     0,     0],
        ],
    }, -0.4),
    # right end of a smile (mirror image)
    "smile_end_r": (1 / 3, {
        "ridge_d45": [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "ridge_h": [
            [0,     0,     0,     0, 0],
            [0,     0,     0,     0, 0],
            [0,     0,     0,     0, 0],
            [1 / 3, 1 / 3, 1 / 3, 0, 0],
            [0,     0,     0,     0, 0],
        ],
    }, -0.4),
    # left end of a frown: the lip line comes in flat from the right and drops down-left
    #   . . - - -
    #   . / . . .
    #   / . . . .
    "frown_end_l": (1 / 3, {
        "ridge_d45": [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "ridge_h": [
            [0, 0, 0,     0,     0],
            [0, 0, 1 / 3, 1 / 3, 1 / 3],
            [0, 0, 0,     0,     0],
            [0, 0, 0,     0,     0],
            [0, 0, 0,     0,     0],
        ],
    }, -0.4),
    # right end of a frown (mirror image)
    "frown_end_r": (1 / 3, {
        "ridge_d135": [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ],
        "ridge_h": [
            [0,     0,     0,     0, 0],
            [1 / 3, 1 / 3, 1 / 3, 0, 0],
            [0,     0,     0,     0, 0],
            [0,     0,     0,     0, 0],
            [0,     0,     0,     0, 0],
        ],
    }, -0.4),
    # an arched brow: flat on top, sloping down at both ends. The slopes carry most of the
    # weight so a straight line does not pass for an arch.
    #   . - - - .
    #   / . . . \
    "arch": (1 / 3, {
        "ridge_h": [
            [0, 0,     0,     0,     0],
            [0, 1 / 3, 1 / 3, 1 / 3, 0],
            [0, 0,     0,     0,     0],
            [0, 0,     0,     0,     0],
            [0, 0,     0,     0,     0],
        ],
        "ridge_d45": [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "ridge_d135": [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    }, -0.4),
}

# --- Stage 3: face zones on the 25x25 maps (one cell = 4x4 original pixels) --------------
# Each zone is a list of boxes (r0, r1, c0, c1), inclusive. L/R mean image left/right.
ZONES = {
    "forehead":     [(0, 2, 6, 18)],                   # far enough up to miss the brows
    "brows_up":     [(3, 5, 3, 9), (3, 5, 15, 21)],    # where raised brows end up
    "brows_rest":   [(6, 6, 3, 9), (6, 6, 15, 21)],    # brow line of the average face
    "brow_l":       [(4, 7, 3, 9)],
    "brow_r":       [(4, 7, 15, 21)],
    "eyes":         [(7, 9, 2, 9), (7, 9, 15, 22)],
    "glabella":     [(4, 8, 10, 14)],                  # between the brows
    "nose_bridge":  [(10, 12, 10, 14)],                # clear of the inner eye corners
    "upper_lip":    [(15, 15, 9, 15)],                 # under the nostrils, over the lip
    "mouth":        [(15, 21, 6, 18)],
    "mouth_center": [(15, 21, 9, 15)],
    "corner_l":     [(15, 20, 4, 10)],
    "corner_r":     [(15, 20, 14, 20)],
}

# --- Stage 4: action units (FACS) ---------------------------------------------------------
# Each AU: (bias, terms). AU = ReLU(bias + sum of coef * pool(map, zone)), pool = "max" or
# "mean" over the zone, map = a conv2 part or a pooled conv1 map. Opposing cues (smile vs
# frown ends, tent vs V brows) subtract each other so a resting face lands near zero.
# Two planned AUs were dropped after the synthetic checks: mouth stretch (AU20) could not be
# told apart from ordinary mouth width, and pressed lips (AU23/24) looked the same as a
# relaxed closed mouth.
AUS = {
    # AU1+2: the brow line moved up and arched, the forehead creased
    "brow_raise": (0.0, [
        (+1.0, "mean", "ridge_h", "brows_up"),
        (-1.0, "mean", "ridge_h", "brows_rest"),
        (+0.5, "max", "arch", "brows_up"),
        (+0.5, "max", "wrinkles", "forehead"),
    ]),
    # AU1: inner brow ends lifted, brows slant like a tent  /  \
    "brows_inner_up": (0.0, [
        (+2.0, "mean", "ridge_d45", "brow_l"),
        (-2.0, "mean", "ridge_d135", "brow_l"),
        (+2.0, "mean", "ridge_d135", "brow_r"),
        (-2.0, "mean", "ridge_d45", "brow_r"),
    ]),
    # AU4: brows pulled down and together, slanting into a V  \  /  with creases between
    "brow_lower": (0.0, [
        (+2.0, "mean", "ridge_d135", "brow_l"),
        (-2.0, "mean", "ridge_d45", "brow_l"),
        (+2.0, "mean", "ridge_d45", "brow_r"),
        (-2.0, "mean", "ridge_d135", "brow_r"),
        (+1.0, "max", "furrow", "glabella"),
    ]),
    # AU5: eyes opened past a thin line: no long slit, no thin bright strip between lids
    "eyes_wide": (0.9, [
        (-1.0, "max", "slit", "eyes"),
        (-1.0, "mean", "bright_h", "eyes"),
    ]),
    # AU6/7: eyes squeezed to a slit by raised cheeks or tightened lids
    "eyes_narrow": (-1.2, [
        (+2.0, "max", "slit", "eyes"),
    ]),
    # AU9
    "nose_wrinkle": (0.0, [
        (+2.0, "max", "wrinkles", "nose_bridge"),
    ]),
    # AU10: the top edge of the upper lip pulled up toward the nose
    "upper_lip_raise": (0.0, [
        (+2.0, "max", "edge_top", "upper_lip"),
    ]),
    # AU12. Frown evidence only counts half: the nose-to-mouth folds of a smile form the
    # same shape as a frown's end.
    "lip_corners_up": (-0.2, [
        (+1.0, "max", "smile_end_l", "corner_l"),
        (+1.0, "max", "smile_end_r", "corner_r"),
        (-0.5, "max", "frown_end_l", "corner_l"),
        (-0.5, "max", "frown_end_r", "corner_r"),
    ]),
    # AU15
    "lip_corners_down": (-0.2, [
        (+1.0, "max", "frown_end_l", "corner_l"),
        (+1.0, "max", "frown_end_r", "corner_r"),
        (-1.0, "max", "smile_end_l", "corner_l"),
        (-1.0, "max", "smile_end_r", "corner_r"),
    ]),
    # AU25 with teeth showing
    "teeth": (0.0, [
        (+2.0, "max", "teeth", "mouth"),
    ]),
    # AU26/27
    "mouth_open": (-0.1, [
        (+1.0, "max", "cavity", "mouth_center"),
    ]),
}

# --- Stage 5: fc layer, action units -> emotion logits (EMFACS prototypes) ----------------
# Column order follows CLASSES. Neutral loses half a unit per unit of any AU, so it wins only
# when the face is quiet; that and its bias are the only places neutral gets evidence.
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

SOFTMAX_TEMPERATURE = 1.0
