"""
Script to create video_index.csv from raw datasets.
This indexes all videos and image sequences for processing.
"""
import os
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATASETS = PROJECT_ROOT / "datasets" / "raw"
VIDEO_INDEX = PROJECT_ROOT / "video_index.csv"

def scan_le2i_dataset():
    """Scan Le2i dataset (videos in fall/ and normal/ folders)."""
    entries = []
    le2i_path = RAW_DATASETS / "le2i"
    
    if not le2i_path.exists():
        return entries
    
    # Scan fall videos
    fall_path = le2i_path / "fall"
    if fall_path.exists():
        for chute_dir in sorted(fall_path.iterdir()):
            if chute_dir.is_dir():
                for video_file in sorted(chute_dir.glob("*.mp4")) + sorted(chute_dir.glob("*.avi")):
                    entries.append({
                        "video_path": str(video_file),
                        "dataset": "le2i",
                        "label": "fall"
                    })
    
    # Scan normal videos
    normal_path = le2i_path / "normal"
    if normal_path.exists():
        for scene_dir in sorted(normal_path.iterdir()):
            if scene_dir.is_dir():
                for video_file in sorted(scene_dir.glob("*.mp4")) + sorted(scene_dir.glob("*.avi")):
                    entries.append({
                        "video_path": str(video_file),
                        "dataset": "le2i",
                        "label": "normal"
                    })
    
    return entries

def scan_urfd_dataset():
    """Scan URFD dataset (image sequences in adl-*/ and fall-*/ folders)."""
    entries = []
    
    for cam in ["urfd_cam0", "urfd_cam1"]:
        cam_path = RAW_DATASETS / cam
        if not cam_path.exists():
            continue
        
        # Scan ADL (normal activities)
        for adl_dir in sorted(cam_path.glob("adl-*-cam*-rgb")):
            if adl_dir.is_dir():
                entries.append({
                    "video_path": str(adl_dir),
                    "dataset": cam,
                    "label": "normal"
                })
        
        # Scan fall sequences
        for fall_dir in sorted(cam_path.glob("fall-*-cam*-rgb")):
            if fall_dir.is_dir():
                entries.append({
                    "video_path": str(fall_dir),
                    "dataset": cam,
                    "label": "fall"
                })
    
    return entries

def main():
    print("Scanning datasets...")
    
    all_entries = []
    all_entries.extend(scan_le2i_dataset())
    all_entries.extend(scan_urfd_dataset())
    
    print(f"Found {len(all_entries)} entries")
    
    # Count by label
    label_counts = {}
    for entry in all_entries:
        label = entry["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"Label distribution: {label_counts}")
    
    # Write CSV
    with open(VIDEO_INDEX, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "dataset", "label"])
        writer.writeheader()
        writer.writerows(all_entries)
    
    print(f"Created {VIDEO_INDEX} with {len(all_entries)} entries")

if __name__ == "__main__":
    main()

