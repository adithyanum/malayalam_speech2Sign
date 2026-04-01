"""
extract_landmarks.py
────────────────────────────────────────────────────────────────
Extracts 21-point hand landmark sequences from a trimmed ISL video
and saves them as a .npy file for playback in the renderer.

Usage:
    python src/extract_landmarks.py --video data/visuals/cat_01.mp4 --label cat --hands 2
    python src/extract_landmarks.py --video data/visuals/me_01.mp4  --label me  --hands 1

Output:
    data/landmarks/cat_landmarks.npy   shape: (n_frames, 2, 21, 3)
    data/landmarks/me_landmarks.npy    shape: (n_frames, 1, 21, 3)
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from scipy.ndimage import gaussian_filter1d

mp_hands = mp.solutions.hands

LANDMARK_DIR  = "data/landmarks"
SMOOTH_SIGMA  = 2.0
ACTIVE_THRESH = 0.01

os.makedirs(LANDMARK_DIR, exist_ok=True)


def extract(video_path: str, label: str, num_hands: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n📹 Video    : {video_path}")
    print(f"   Frames   : {total}  |  FPS: {fps:.1f}  |  Duration: {total/fps:.2f}s")
    print(f"   Label    : {label}")
    print(f"   Hands    : {num_hands}")

    sequence = []
    detected = 0
    skipped  = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=num_hands,          # ← only look for as many hands as the sign needs
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result    = hands.process(rgb)

            # shape: (num_hands, 21, 3) — inactive slots stay zero
            frame_lms = np.zeros((num_hands, 21, 3), dtype=np.float32)

            if result.multi_hand_landmarks:
                detected += 1
                for i, hand_lms in enumerate(result.multi_hand_landmarks[:num_hands]):
                    frame_lms[i] = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                        dtype=np.float32
                    )
            else:
                skipped += 1

            sequence.append(frame_lms)

    cap.release()

    if detected == 0:
        print("❌ No hand detected in any frame.")
        return

    seq_array = np.array(sequence, dtype=np.float32)  # (n_frames, num_hands, 21, 3)

    # Smooth only within active frames per hand slot
    print(f"   Smoothing active tracks (sigma={SMOOTH_SIGMA})...")
    for hand_idx in range(num_hands):
        active_mask = (
            (seq_array[:, hand_idx, 0, 0] > ACTIVE_THRESH) &
            (seq_array[:, hand_idx, 0, 1] > ACTIVE_THRESH)
        )
        active_indices = np.where(active_mask)[0]
        if len(active_indices) < 3:
            continue
        start, end = active_indices[0], active_indices[-1] + 1
        for lm_idx in range(21):
            for axis in range(3):
                seq_array[start:end, hand_idx, lm_idx, axis] = gaussian_filter1d(
                    seq_array[start:end, hand_idx, lm_idx, axis],
                    sigma=SMOOTH_SIGMA
                )

    output_path = os.path.join(LANDMARK_DIR, f"{label}_landmarks.npy")
    np.save(output_path, seq_array)

    pct       = 100 * detected / total
    avg_hands = np.mean([
        sum(seq_array[i, h, 0, 0] > ACTIVE_THRESH for h in range(num_hands))
        for i in range(len(seq_array))
    ])

    print(f"\n✅ Extraction complete!")
    print(f"   Detected       : {detected}/{total} frames ({pct:.1f}%)")
    print(f"   Skipped        : {skipped} frames (no hand)")
    print(f"   Avg hands/frame: {avg_hands:.2f}")
    print(f"   Saved          : {output_path}")
    print(f"   Shape          : {seq_array.shape}  (frames × {num_hands} hand(s) × 21 landmarks × xyz)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True,  help="Path to trimmed ISL video")
    parser.add_argument("--label",  required=True,  help="Word label e.g. cat, me, house")
    parser.add_argument("--hands",  type=int, default=2,
                        help="Number of hands in this sign: 1 or 2 (default: 2)")
    parser.add_argument("--smooth", type=float, default=SMOOTH_SIGMA)
    args = parser.parse_args()
    SMOOTH_SIGMA = args.smooth
    extract(args.video, args.label, args.hands)