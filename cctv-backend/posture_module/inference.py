"""
Real-time posture analysis with video feed and alerts.
Detects: normal (standing), fall, lying/crawling
"""
import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
import random
from pathlib import Path
import time
from datetime import datetime

from model import PostureLSTM, PostureClassifier
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "posture_model_best.pth"

# Constants
NUM_LANDMARKS = 33
SEQUENCE_LENGTH = 60
ALERT_THRESHOLD = 0.7  # Confidence threshold for alerts
ALERT_DURATION = 2.0  # Seconds of abnormal posture before alert
MIN_DET_CONF = 0.3  # Lowered for pose detection
MIN_VISIBILITY = 0.25  # Visibility cutoff per-frame (mean across 33 lms)
MIN_VALID_FRAMES = 5  # Lowered for testing to allow faster buffer fill
HORIZ_RATIO_THRESH = 0.6  # Height/width ratio to consider posture horizontal
MIN_GROUND_TIME = 1.5  # Seconds horizontal after fall prob spike to confirm
COOLDOWN_TIME = 5.0  # Seconds after alert before re-arming
PERSON_CONF_THRESH = 0.3  # Detector confidence for person
MIN_BOX_SIZE = 0.03  # Min box size relative to frame (either height or width)
CROP_SIZE = 320  # Square crop size for pose input

# Geometry-only fall confirmation: N consecutive horizontal frames
FALL_HORIZONTAL_FRAMES_CONFIRM = 20
RECOVERY_FRAMES = 30  # Number of upright frames to consider recovered

FPS = 30


def iou(boxA, boxB):
    """Compute IoU between two boxes (x1,y1,x2,y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


class PostureAnalyzer:
    """Real-time posture analyzer with alert system."""
    
    def __init__(self, model_path=None, camera_id=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        if model_path is None:
            model_path = MODEL_PATH
        
        if not Path(model_path).exists():
            print(f"WARNING: Model not found at {model_path}")
            print("Please train the model first using train.py")
            self.model = None
        else:
            self.classifier = PostureClassifier(model_path, self.device)
            self.model = self.classifier.model

        # Person detector (COCO-pretrained)
        self.detector = YOLO("yolov8n.pt")
        
        # MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Class indices
        self.class_names = getattr(self, "classifier", None) and self.classifier.class_names or ["normal", "fall", "lying"]
        self.fall_index = self.class_names.index("fall") if "fall" in self.class_names else 1
        self.lying_index = self.class_names.index("lying") if "lying" in self.class_names else 2
        print("Loaded class names:", self.class_names)
        print("fall_index:", self.fall_index, "lying_index:", self.lying_index)
        
        # Per-track buffers and state
        self.tracks = {}  # track_id -> dict of buffers/state
        self.next_track_id = 1
        
        # Camera
        self.camera_id = camera_id
        self.cap = None
        
    def normalize_skeleton(self, landmarks):
        """Normalize pose landmarks (same as in extract_pose_sequences.py)."""
        # hip center
        left_hip = landmarks[23][:2]
        right_hip = landmarks[24][:2]
        center = (left_hip + right_hip) / 2.0
        
        coords = landmarks[:, :2] - center
        
        # torso height
        left_shoulder = landmarks[11][:2]
        right_shoulder = landmarks[12][:2]
        shoulders_center = (left_shoulder + right_shoulder) / 2.0
        hips_center = (left_hip + right_hip) / 2.0
        
        torso_height = np.linalg.norm(shoulders_center - hips_center) + 1e-6
        coords = coords / torso_height
        
        return coords.flatten().astype(np.float32)
    
    def _is_pose_confident(self, landmarks):
        """Return True if the pose landmarks are reliable enough to use."""
        visibilities = landmarks[:, 2]
        avg_visibility = np.mean(visibilities)

        # Check a few critical joints explicitly
        key_indices = [11, 12, 23, 24]  # shoulders and hips
        key_visibility = np.mean(visibilities[key_indices])

        return avg_visibility >= MIN_VISIBILITY and key_visibility >= MIN_VISIBILITY

    def is_horizontal(self, pose_landmarks):
        """Heuristic: determine if the person is horizontal/on-ground using h/w."""
        if not pose_landmarks:
            return False
        coords = np.array([[lm.x, lm.y] for lm in pose_landmarks.landmark])
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        h = y_max - y_min
        w = x_max - x_min
        ratio = h / (w + 1e-6)
        return ratio <= HORIZ_RATIO_THRESH

    def match_tracks(self, detections, now_ts):
        """Assign detections to existing tracks using IoU; create new tracks as needed."""
        assignments = []
        used_tracks = set()

        for det in detections:
            best_track = None
            best_iou = 0.0
            for tid, t in self.tracks.items():
                iou_val = iou(det["bbox"], t["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_track = tid
            if best_track is not None and best_iou >= 0.5 and best_track not in used_tracks:
                assignments.append((best_track, det))
                used_tracks.add(best_track)
            else:
                # new track
                tid = self.next_track_id
                self.next_track_id += 1
                self._ensure_track(tid, det["bbox"], now_ts)
                assignments.append((tid, det))
                used_tracks.add(tid)

        # Remove stale tracks with state-aware timeout
        stale = []
        for tid, t in self.tracks.items():
            timeout = 5.0
            if t.get("state") in ("FALLING", "ON_GROUND", "FALLEN"):
                timeout = 8.0
            if now_ts - t.get("last_seen", 0) > timeout:
                stale.append(tid)
        for tid in stale:
            self.tracks.pop(tid, None)

        # Update bbox/last_seen for matched tracks
        for tid, det in assignments:
            self._ensure_track(tid, det["bbox"], now_ts)

        return assignments

    @staticmethod
    def _horizontal_from_coords(coords):
        if not coords:
            return False
        arr = np.array(coords)
        x_min, y_min = arr.min(axis=0)
        x_max, y_max = arr.max(axis=0)
        h = y_max - y_min
        w = x_max - x_min
        ratio = h / (w + 1e-6)
        return ratio <= HORIZ_RATIO_THRESH

    @staticmethod
    def _horizontal_from_bbox(bbox):
        """Use bbox aspect ratio as a backup horizontal check."""
        x1, y1, x2, y2 = bbox
        h = abs(y2 - y1)
        w = abs(x2 - x1)
        ratio = h / (w + 1e-6)
        return ratio <= HORIZ_RATIO_THRESH

    def detect_people(self, frame):
        """Run person detector and return list of bboxes with scores."""
        results = self.detector.predict(frame, verbose=False)
        bboxes = []
        if not results:
            return bboxes
        res = results[0]
        h, w, _ = frame.shape
        for box, cls, score in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            if int(cls) != 0:  # person class in COCO
                continue
            if score < PERSON_CONF_THRESH:
                continue
            x1, y1, x2, y2 = box
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            if bw < MIN_BOX_SIZE or bh < MIN_BOX_SIZE:
                continue
            bboxes.append({"bbox": (float(x1), float(y1), float(x2), float(y2)), "score": float(score)})
        return bboxes

    def pose_from_crop(self, frame, bbox):
        """Crop person, run pose on the crop, return pose_vec, landmarks_abs, is_valid."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w, _ = frame.shape

        # Clamp bbox
        x1 = max(0, min(w - 1, x1))
        x2 = max(1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(1, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            zero_vec = np.zeros(2 * NUM_LANDMARKS, dtype=np.float32)
            return zero_vec, None, False

        # Enlarge bbox by 20%
        pad = 0.2
        bw = x2 - x1
        bh = y2 - y1
        x1 = int(max(0, x1 - pad * bw))
        x2 = int(min(w, x2 + pad * bw))
        y1 = int(max(0, y1 - pad * bh))
        y2 = int(min(h, y2 + pad * bh))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            zero_vec = np.zeros(2 * NUM_LANDMARKS, dtype=np.float32)
            return zero_vec, None, False

        # Aspect-ratio preserve resize (no square stretch)
        ratio = CROP_SIZE / max(crop.shape[0], crop.shape[1])
        new_w = int(crop.shape[1] * ratio)
        new_h = int(crop.shape[0] * ratio)
        crop_resized = cv2.resize(crop, (new_w, new_h))

        img_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            landmarks = np.array([[p.x, p.y, p.visibility] for p in lm])

            # Map landmarks back to original frame coords for drawing (debug)
            coords_debug = []
            for p in lm:
                cx = p.x * new_w
                cy = p.y * new_h
                ox = x1 + cx
                oy = y1 + cy
                coords_debug.append((ox, oy))

            if landmarks.shape[0] == NUM_LANDMARKS and self._is_pose_confident(landmarks):
                pose_vec = self.normalize_skeleton(landmarks)
                return pose_vec, coords_debug, True
            # Low confidence but return coords for drawing
            zero_vec = np.zeros(2 * NUM_LANDMARKS, dtype=np.float32)
            return zero_vec, coords_debug, False

        zero_vec = np.zeros(2 * NUM_LANDMARKS, dtype=np.float32)
        return zero_vec, None, False

    def _ensure_track(self, track_id, bbox, now_ts):
        """Create track if missing."""
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                "pose_sequence": deque(maxlen=SEQUENCE_LENGTH),
                "valid_sequence": deque(maxlen=SEQUENCE_LENGTH),
                "state": "NORMAL",
                "fall_start_time": None,
                "last_alert_time": 0.0,
                "fall_frames": 0,
                "horizontal_frames": 0,
                "upright_frames": 0,
                "recovery_frames": 0,  # legacy, kept for safety
                "last_seen": now_ts,
                "bbox": bbox,
                "color": (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
                # For event-driven debug logging
                "debug_prev_state": None,
                "debug_prev_class": None,
                "debug_prev_horizontal": None,
            }
        else:
            self.tracks[track_id]["bbox"] = bbox
            self.tracks[track_id]["last_seen"] = now_ts

    def _update_track_buffers(self, track_id, pose_vec, is_valid):
        t = self.tracks[track_id]
        t["pose_sequence"].append(pose_vec)
        t["valid_sequence"].append(1 if is_valid else 0)

    def _predict_track(self, track_id):
        t = self.tracks[track_id]
        seq = np.array(list(t["pose_sequence"]))
        valid_mask = np.array(list(t["valid_sequence"]), dtype=np.float32)
        if len(seq) < MIN_VALID_FRAMES:
            return "no_person", 0.0, np.zeros(3, dtype=np.float32)

        valid_count = int(valid_mask.sum())
        if seq.shape[0] < SEQUENCE_LENGTH:
            pad = SEQUENCE_LENGTH - seq.shape[0]
            seq = np.pad(seq, ((0, pad), (0, 0)), mode="constant")
            valid_mask = np.pad(valid_mask, (0, pad), mode="constant")
        elif seq.shape[0] > SEQUENCE_LENGTH:
            start = (seq.shape[0] - SEQUENCE_LENGTH) // 2
            seq = seq[start:start + SEQUENCE_LENGTH]
            valid_mask = valid_mask[start:start + SEQUENCE_LENGTH]
            valid_count = int(valid_mask.sum())

        if valid_count < MIN_VALID_FRAMES:
            return "no_person", 0.0, np.zeros(3, dtype=np.float32)

        seq_tensor = torch.from_numpy(seq).float()
        if seq_tensor.dim() == 2:
            seq_tensor = seq_tensor.unsqueeze(0)
        seq_tensor = seq_tensor.to(self.device)
        with torch.no_grad():
            probs = self.model.predict_proba(seq_tensor).cpu().numpy()[0]
        class_idx = int(np.argmax(probs))
        class_name = self.classifier.class_names[class_idx]
        confidence = float(probs[class_idx])
        return class_name, confidence, probs

    def _update_fall_state_track(self, track_id, horizontal):
        """Simplified geometry-based fall state machine per track."""
        t = self.tracks[track_id]
        current_time = time.time()
        alerts = []

        # Update counters based on geometry only
        if horizontal:
            t["horizontal_frames"] = min(t.get("horizontal_frames", 0) + 1, 1000)
            t["upright_frames"] = 0
        else:
            t["upright_frames"] = min(t.get("upright_frames", 0) + 1, 1000)
            t["horizontal_frames"] = 0

        # NORMAL -> FALLEN: sustained horizontal
        if t["state"] == "NORMAL":
            if t["horizontal_frames"] >= FALL_HORIZONTAL_FRAMES_CONFIRM:
                print(f"[INFO] track={track_id} FALL CONFIRMED (bbox/pose horizontal)")
                t["state"] = "FALLEN"
                t["last_alert_time"] = current_time
                alerts.append(f"FALL DETECTED (track_id={track_id})")

        # FALLEN -> NORMAL: sustained upright
        elif t["state"] == "FALLEN":
            if t["upright_frames"] >= RECOVERY_FRAMES:
                print(f"[INFO] track={track_id} RECOVERED after fall")
                t["state"] = "NORMAL"
                t["horizontal_frames"] = 0
                t["upright_frames"] = 0

        return alerts

    def _update_lying_alert_track(self, track_id, predicted_class, confidence):
        """Per-track lying alert with duration gating."""
        current_time = time.time()
        t = self.tracks[track_id]
        lying_state = t.setdefault("lying_state", {"active": False, "start_time": None})
        alerts = []

        if predicted_class == "lying" and confidence >= ALERT_THRESHOLD:
            if not lying_state["active"]:
                lying_state["start_time"] = current_time
                lying_state["active"] = True
            elif current_time - lying_state["start_time"] >= ALERT_DURATION:
                alerts.append(f"PERSON LYING DOWN (track_id={track_id})")
        else:
            lying_state["active"] = False

        return alerts

    # --- legacy single-person API (unused in multi-track path, left for compatibility) ---

    def process_frame(self, frame):
        """Process a frame and return (pose_vec, landmarks_to_draw, is_valid).

        Low-visibility poses are replaced with zeros and marked invalid so they
        cannot trigger false alerts or dominate the sliding window.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            landmarks = np.array([[p.x, p.y, p.visibility] for p in lm])
            
            if landmarks.shape[0] == NUM_LANDMARKS and self._is_pose_confident(landmarks):
                pose_vec = self.normalize_skeleton(landmarks)
                return pose_vec, results.pose_landmarks, True
            # Low-confidence detection: return zero vector
            zero_vec = np.zeros(2 * NUM_LANDMARKS, dtype=np.float32)
            return zero_vec, None, False

        # No landmarks detected
        zero_vec = np.zeros(2 * NUM_LANDMARKS, dtype=np.float32)
        return zero_vec, None, False
    
    def predict_posture(self):
        """Predict posture from current sequence buffer (legacy single-person)."""
        if self.model is None or not hasattr(self, "pose_sequence") or len(self.pose_sequence) < MIN_VALID_FRAMES:
            return "no_person", 0.0, np.zeros(3, dtype=np.float32)
        
        sequence = np.array(list(self.pose_sequence))
        valid_mask = np.array(list(self.valid_sequence), dtype=np.float32)
        valid_count = int(valid_mask.sum())
        
        if sequence.shape[0] < SEQUENCE_LENGTH:
            pad_length = SEQUENCE_LENGTH - sequence.shape[0]
            sequence = np.pad(sequence, ((0, pad_length), (0, 0)), mode="constant")
            valid_mask = np.pad(valid_mask, (0, pad_length), mode="constant")
        elif sequence.shape[0] > SEQUENCE_LENGTH:
            start = (sequence.shape[0] - SEQUENCE_LENGTH) // 2
            sequence = sequence[start:start + SEQUENCE_LENGTH]
            valid_mask = valid_mask[start:start + SEQUENCE_LENGTH]
            valid_count = int(valid_mask.sum())
        
        if valid_count < MIN_VALID_FRAMES:
            return "no_person", 0.0, np.zeros(3, dtype=np.float32)
        
        seq_tensor = torch.from_numpy(sequence).float()
        if seq_tensor.dim() == 2:
            seq_tensor = seq_tensor.unsqueeze(0)
        seq_tensor = seq_tensor.to(self.device)
        with torch.no_grad():
            probs = self.model.predict_proba(seq_tensor).cpu().numpy()[0]
        class_idx = int(np.argmax(probs))
        class_name = self.classifier.class_names[class_idx]
        confidence = float(probs[class_idx])
        return class_name, confidence, probs
    
    def update_fall_state(self, probs, confidence, horizontal):
        """Legacy single-person fall state (unused)."""
        return []

    def update_lying_alert(self, predicted_class, confidence):
        """Legacy single-person lying state (unused)."""
        return []
    
    def draw_tracks(self, frame, track_results):
        """Draw per-track bboxes, keypoints, and status."""
        for tr in track_results:
            tid = tr["track_id"]
            bbox = tr["bbox"]
            coords = tr["coords"]
            predicted_class = tr["predicted_class"]
            confidence = tr["confidence"]
            alerts = tr["alerts"]
            color = tr["color"]

            # Use robust fallback if track somehow vanished
            state = self.tracks.get(tid, {}).get("state", "UNKNOWN")

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if coords:
                for (cx, cy) in coords:
                    cv2.circle(frame, (int(cx), int(cy)), 2, color, -1)

            # Show STATE as the main label, plus model class for debugging
            status_text = f"id={tid} {state} | cls={predicted_class} ({confidence:.2f})"
            cv2.putText(
                frame,
                status_text,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            if alerts:
                alert_text = " | ".join(alerts)
                cv2.putText(
                    frame,
                    alert_text,
                    (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        return frame
    
    def run(self, video_source=None):
        """Run real-time analysis."""
        # Open video source
        if video_source is None:
            self.cap = cv2.VideoCapture(self.camera_id)
        elif isinstance(video_source, str):
            self.cap = cv2.VideoCapture(video_source)
        else:
            self.cap = video_source
        
        if not self.cap.isOpened():
            print(f"ERROR: Could not open video source")
            return
        
        print("Starting posture analysis...")
        print("Press 'q' to quit")
        
        total_frames = 0
        frames_with_detections = 0
        frames_with_pose = 0

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                total_frames += 1
                now_ts = time.time()
                detections = self.detect_people(frame)
                if detections:
                    frames_with_detections += 1
                assignments = self.match_tracks(detections, now_ts)
                track_results = []

                for tid, det in assignments:
                    bbox = det["bbox"]
                    pose_vec, coords, is_valid = self.pose_from_crop(frame, bbox)
                    self._ensure_track(tid, bbox, now_ts)
                    self._update_track_buffers(tid, pose_vec, is_valid)

                    predicted_class, confidence, probs = self._predict_track(tid)

                    # Horizontal check: pose OR bbox
                    horiz_pose = self._horizontal_from_coords(coords)
                    horiz_box = self._horizontal_from_bbox(bbox)
                    horizontal = horiz_pose or horiz_box

                    # Event-driven debug logging:
                    # Only log when state/class/horizontal actually changes
                    state = self.tracks[tid]["state"]
                    prev_state = self.tracks[tid].get("debug_prev_state")
                    prev_class = self.tracks[tid].get("debug_prev_class")
                    prev_horizontal = self.tracks[tid].get("debug_prev_horizontal")

                    if (
                        state != prev_state
                        or predicted_class != prev_class
                        or horizontal != prev_horizontal
                    ):
                        valid_len = len(self.tracks[tid]["pose_sequence"])
                        valid_count = int(np.sum(self.tracks[tid]["valid_sequence"]))
                        print(
                            f"[DEBUG] track={tid} state={state} "
                            f"class={predicted_class} conf={confidence:.3f} "
                            f"horizontal={horizontal} seq_len={valid_len} "
                            f"valid_frames={valid_count}"
                        )
                        self.tracks[tid]["debug_prev_state"] = state
                        self.tracks[tid]["debug_prev_class"] = predicted_class
                        self.tracks[tid]["debug_prev_horizontal"] = horizontal

                    alerts = []
                    # Fall purely from geometry + persistence
                    alerts.extend(self._update_fall_state_track(tid, horizontal))
                    # Lying still uses classifier if available
                    if predicted_class != "no_person":
                        alerts.extend(self._update_lying_alert_track(tid, predicted_class, confidence))

                    # Print alerts once per event
                    for a in alerts:
                        print(f"[ALERT] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {a}")

                    track_results.append({
                        "track_id": tid,
                        "bbox": bbox,
                        "coords": coords,
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "alerts": alerts,
                        "color": self.tracks[tid]["color"],
                    })

                if any(tr["coords"] for tr in track_results):
                    frames_with_pose += 1

                # Draw overlay
                frame = self.draw_tracks(frame, track_results)
                
                # Display frame
                cv2.imshow("Posture Analysis", frame)
                
                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.pose.close()
            print(f"Total frames: {total_frames}")
            print(f"Frames with detections: {frames_with_detections}")
            print(f"Frames with pose drawn: {frames_with_pose}")
            print("Analysis stopped")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time posture analysis")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    args = parser.parse_args()
    
    analyzer = PostureAnalyzer(model_path=args.model, camera_id=args.camera)
    analyzer.run(video_source=args.video)

if __name__ == "__main__":
    main()
