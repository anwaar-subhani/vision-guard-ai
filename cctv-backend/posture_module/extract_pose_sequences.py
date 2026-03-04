import os
import csv
import cv2
import numpy as np
import mediapipe as mp

VIDEO_INDEX = "video_index.csv"
OUTPUT_DIR = "datasets/processed/pose_sequences"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
NUM_LANDMARKS = 33

LABEL_MAP = {"normal": 0, "fall": 1, "lying": 2}  # lying unused for now, but reserved

def normalize_skeleton(landmarks):
    """
    landmarks: (33, 3) -> [x, y, visibility]
    Returns flattened normalized pose: (2 * 33,)
    """
    # hip center
    left_hip = landmarks[23][:2]
    right_hip = landmarks[24][:2]
    center = (left_hip + right_hip) / 2.0

    coords = landmarks[:, :2] - center  # subtract center

    # torso height = distance between (shoulders center) and (hips center)
    left_shoulder = landmarks[11][:2]
    right_shoulder = landmarks[12][:2]
    shoulders_center = (left_shoulder + right_shoulder) / 2.0
    hips_center = (left_hip + right_hip) / 2.0

    torso_height = np.linalg.norm(shoulders_center - hips_center) + 1e-6
    coords = coords / torso_height

    return coords.flatten().astype(np.float32)

def process_urfd_sequence_dir(seq_dir, label, dataset):
    """seq_dir contains frames (png/jpg)."""
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = sorted(
        [f for f in os.listdir(seq_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    sequence = []

    for fname in frames:
        fpath = os.path.join(seq_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            landmarks = np.array([[p.x, p.y, p.visibility] for p in lm])
            if landmarks.shape[0] != NUM_LANDMARKS:
                continue
            pose_vec = normalize_skeleton(landmarks)
        else:
            pose_vec = np.zeros(2 * NUM_LANDMARKS, dtype=np.float32)

        sequence.append(pose_vec)

    pose.close()

    if not sequence:
        print(f"[WARN] No frames processed for {seq_dir}")
        return

    X = np.stack(sequence, axis=0)  # (T, F)
    y = LABEL_MAP.get(label, 0)

    base_name = f"{dataset}_" + os.path.basename(seq_dir.rstrip("/\\"))
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}.npz")
    np.savez_compressed(out_path, X=X, y=y)

    print(f"[URFD] Saved {out_path}, shape={X.shape}, label={label}")

def process_le2i_video(video_path, label, dataset):
    """video_path is a file (.avi/.mp4)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERR] Could not open video: {video_path}")
        return

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            landmarks = np.array([[p.x, p.y, p.visibility] for p in lm])
            if landmarks.shape[0] != NUM_LANDMARKS:
                continue
            pose_vec = normalize_skeleton(landmarks)
        else:
            pose_vec = np.zeros(2 * NUM_LANDMARKS, dtype=np.float32)

        sequence.append(pose_vec)

    cap.release()
    pose.close()

    if not sequence:
        print(f"[WARN] No frames processed for {video_path}")
        return

    X = np.stack(sequence, axis=0)
    y = LABEL_MAP.get(label, 0)

    base_name = f"{dataset}_" + os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}.npz")
    np.savez_compressed(out_path, X=X, y=y)

    print(f"[Le2i] Saved {out_path}, shape={X.shape}, label={label}")

def main():
    with open(VIDEO_INDEX, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["video_path"]
            dataset = row["dataset"]
            label = row["label"]

            if not os.path.exists(path):
                print(f"[MISS] {path} does not exist, skipping")
                continue

            # URFD paths are directories of frames
            if dataset.startswith("urfd"):
                if os.path.isdir(path):
                    process_urfd_sequence_dir(path, label, dataset)
                else:
                    print(f"[WARN] Expected dir for URFD, got file: {path}")
            # Le2i paths are video files
            elif dataset == "le2i":
                if os.path.isfile(path):
                    process_le2i_video(path, label, dataset)
                else:
                    print(f"[WARN] Expected file for Le2i, got dir: {path}")
            else:
                print(f"[WARN] Unknown dataset={dataset} for path={path}")

if __name__ == "__main__":
    main()
