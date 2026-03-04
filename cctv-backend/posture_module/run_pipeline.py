"""
Complete pipeline script to process datasets and train the model.
Run this script to go from raw datasets to trained model.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def run_step(step_name, script_path, description):
    """Run a pipeline step."""
    print("\n" + "="*60)
    print(f"Step: {step_name}")
    print(f"Description: {description}")
    print("="*60)
    
    script_full_path = PROJECT_ROOT / script_path
    
    if not script_full_path.exists():
        print(f"ERROR: Script not found: {script_full_path}")
        return False
    
    import subprocess
    result = subprocess.run([sys.executable, str(script_full_path)], cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"ERROR: Step {step_name} failed with return code {result.returncode}")
        return False
    
    print(f"✓ Step {step_name} completed successfully")
    return True

def main():
    print("Posture Analysis Pipeline")
    print("="*60)
    print("This script will:")
    print("1. Create video index from raw datasets")
    print("2. Extract pose sequences from videos")
    print("3. Train the posture classification model")
    print("\nPress Ctrl+C to cancel at any time\n")
    
    try:
        # Step 1: Create video index
        if not run_step(
            "1/3",
            "posture_module/create_video_index.py",
            "Creating video index from raw datasets"
        ):
            return
        
        # Step 2: Extract pose sequences
        if not run_step(
            "2/3",
            "posture_module/extract_pose_sequences.py",
            "Extracting pose sequences from videos (this may take a while)"
        ):
            return
        
        # Step 3: Train model
        if not run_step(
            "3/3",
            "posture_module/train.py",
            "Training the posture classification model"
        ):
            return
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)
        print("\nYou can now run inference with:")
        print("  python posture_module/inference.py --camera 0")
        print("\nOr analyze a video file:")
        print("  python posture_module/inference.py --video path/to/video.mp4")
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()

