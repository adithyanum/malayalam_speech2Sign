"""
sign_renderer.py
────────────────────────────────────────────────────────────────
Renders ISL hand skeleton animation on a black canvas.

Standalone:
    python src/sign_renderer.py --label cat
"""

import cv2
import numpy as np
import os
import threading

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

FINGERTIPS = {4, 8, 12, 16, 20}
WRIST      = 0

CANVAS_W = 640
CANVAS_H = 580
HUD_H    = 70
PAD      = 35
FPS      = 30

HAND_COLORS = [
    {"line": (20, 200, 160),  "knuckle": (60, 180, 80),  "tip": (80, 220, 255), "wrist": (255, 180, 0)},
    {"line": (220, 80, 200),  "knuckle": (180, 60, 180), "tip": (255, 140, 80), "wrist": (255, 220, 60)},
]

# A hand is considered active only if its wrist landmark is clearly non-zero
ACTIVE_THRESHOLD = 0.01


def _is_active(hand_lms: np.ndarray) -> bool:
    """
    Check if a hand slot is genuinely detected.
    We check the wrist (landmark 0) — if it's near (0,0) it's just padding.
    """
    return float(hand_lms[0, 0]) > ACTIVE_THRESHOLD and float(hand_lms[0, 1]) > ACTIVE_THRESHOLD


def _compute_transform(sequence: np.ndarray):
    """
    Tight bounding box using only genuinely active hand landmarks.
    sequence: (n_frames, 2, 21, 3)
    """
    xs, ys = [], []
    for frame in sequence:
        for hand_idx in range(sequence.shape[1]):
            if _is_active(frame[hand_idx]):
                xs.extend(frame[hand_idx, :, 0].tolist())
                ys.extend(frame[hand_idx, :, 1].tolist())

    if not xs:
        return 1.0, 0.0, 0.0, PAD, HUD_H + PAD

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_range = x_max - x_min if x_max > x_min else 1e-6
    y_range = y_max - y_min if y_max > y_min else 1e-6

    draw_w = CANVAS_W - 2 * PAD
    draw_h = CANVAS_H - HUD_H - 2 * PAD

    scale    = min(draw_w / x_range, draw_h / y_range)
    scaled_w = x_range * scale
    scaled_h = y_range * scale
    x_off    = PAD + (draw_w - scaled_w) / 2
    y_off    = HUD_H + PAD + (draw_h - scaled_h) / 2

    return scale, x_min, y_min, x_off, y_off


def _to_px(lm, scale, x_min, y_min, x_off, y_off):
    px = int((lm[0] - x_min) * scale + x_off)
    py = int((lm[1] - y_min) * scale + y_off)
    return (px, py)


def _draw_hand(canvas, pts, colors):
    for (a, b) in CONNECTIONS:
        cv2.line(canvas, pts[a], pts[b], colors["line"], 2, lineType=cv2.LINE_AA)
    for idx, (px, py) in enumerate(pts):
        if idx == WRIST:
            color, r = colors["wrist"], 9
        elif idx in FINGERTIPS:
            color, r = colors["tip"], 7
        else:
            color, r = colors["knuckle"], 5
        cv2.circle(canvas, (px, py), r,     color,           -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (px, py), r + 1, (255, 255, 255),  1, lineType=cv2.LINE_AA)


def _draw_frame(frame_lms, transform, word_label, confidence,
                frame_idx, total_frames):
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    scale, x_min, y_min, x_off, y_off = transform

    for hand_idx in range(len(frame_lms)):
        hand = frame_lms[hand_idx]
        if not _is_active(hand):
            continue
        pts = [_to_px(lm, scale, x_min, y_min, x_off, y_off) for lm in hand]
        _draw_hand(canvas, pts, HAND_COLORS[hand_idx])

    # HUD
    cv2.putText(canvas, word_label.upper(),
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(canvas, f"{int(confidence * 100)}% confidence",
                (20, 62), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (120, 120, 120), 1, lineType=cv2.LINE_AA)

    # Progress bar
    bx, by, bh = 20, CANVAS_H - 16, 5
    bw   = CANVAS_W - 40
    prog = int(bw * (frame_idx / max(total_frames - 1, 1)))
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (45, 45, 45), -1)
    cv2.rectangle(canvas, (bx, by), (bx + prog, by + bh), (20, 200, 160), -1)
    cv2.putText(canvas, "Q to close",
                (CANVAS_W - 85, CANVAS_H - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (65, 65, 65), 1)

    return canvas


def play_sign(label: str, confidence: float = 1.0,
              landmark_dir: str = "data/landmarks", loop: bool = True):

    npy_path = os.path.join(landmark_dir, f"{label}_landmarks.npy")
    if not os.path.exists(npy_path):
        print(f"⚠️  No landmark file for '{label}' — run extract_landmarks.py first.")
        return

    sequence = np.load(npy_path)

    # Handle old single-hand format (n_frames, 21, 3)
    if sequence.ndim == 3:
        padded = np.zeros((len(sequence), 2, 21, 3), dtype=np.float32)
        padded[:, 0] = sequence
        sequence = padded

    total_frames = len(sequence)
    delay_ms     = int(1000 / FPS)
    transform    = _compute_transform(sequence)

    window_name = f"ISL Sign"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_idx = 0
    while True:
        canvas = _draw_frame(sequence[frame_idx], transform, label, confidence,
                             frame_idx, total_frames)
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord('q'), 27):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        frame_idx += 1
        if frame_idx >= total_frames:
            if loop:
                frame_idx = 0
            else:
                break

    cv2.destroyWindow(window_name)


def play_sign_async(label: str, confidence: float = 1.0,
                    landmark_dir: str = "data/landmarks"):
    t = threading.Thread(
        target=play_sign,
        args=(label, confidence, landmark_dir, True),
        daemon=True
    )
    t.start()
    return t


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",        required=True)
    parser.add_argument("--landmark_dir", default="data/landmarks")
    parser.add_argument("--no-loop",      action="store_true")
    args = parser.parse_args()
    print(f"▶  Playing: {args.label}  |  Q or ESC to close")
    play_sign(args.label, confidence=1.0,
              landmark_dir=args.landmark_dir,
              loop=not args.no_loop)