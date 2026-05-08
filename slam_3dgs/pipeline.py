"""
Automated training + evaluation pipeline for 3DGS SLAM.
Runs all steps for each track in sequence.

Usage:
    python pipeline.py                        # run all 3 tracks
    python pipeline.py --tracks 1 2           # run specific tracks
    python pipeline.py --tracks 1 --skip_train  # skip training (resume from checkpoint)

What it does per track:
    1. test_camera_poses.py → output/track{N}/visualization.ply
    2. generate_sky_mask.py → itri58_colored_pcd_t{N}/sky_masks/
    3. train.py (if exists) or train.py → output/track{N}/gaussian_reconstruction.ply
    4. generate_test_pose.py → track{N}/test_pose_list.txt
    5. generate_submission.py → track{N}/test_submission/*.png (uses trained model, not visualization.ply)
    6. Zip step → track{N}/submission.zip
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
BASE = SCRIPT_DIR.parent.parent / "slam"
SLAM_LIDAR = BASE / "slam_lidar"


def run(script_name: str, *args):
    """Run a Python script in the slam_3dgs directory."""
    script = SCRIPT_DIR / script_name
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    cmd = [sys.executable, str(script)] + [str(a) for a in args]
    print(f"\n{'='*60}")
    print(f"Running: {script_name} {' '.join(str(a) for a in args)}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR))


def make_zip(track_dir: Path, track_num: int):
    """Zip the 30 rendered PNG images and submission.csv."""
    test_submission_dir = track_dir / "test_submission"
    submission_csv = track_dir / "submission.csv"
    zip_path = track_dir / "submission.zip"

    if not test_submission_dir.exists():
        raise FileNotFoundError(f"test_submission/ not found at {test_submission_dir}")
    if not submission_csv.exists():
        raise FileNotFoundError(f"submission.csv not found at {submission_csv}")

    png_files = sorted(test_submission_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {test_submission_dir}")

    print(f"\nCreating {zip_path} with {len(png_files)} images + submission.csv ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in png_files:
            zf.write(img, img.name)
        zf.write(submission_csv, "submission.csv")

    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"Done: {zip_path} ({size_mb:.1f} MB, {len(png_files)} images)")


def run_track(track_num: int, skip_train: bool = False):
    data_dir = SLAM_LIDAR / f"itri58_colored_pcd_t{track_num}"
    track_dir = SCRIPT_DIR / f"track{track_num}"
    output_dir = SCRIPT_DIR / "output" / f"track{track_num}"
    vis_ply = output_dir / "visualization.ply"
    trained_ply = output_dir / "gaussian_reconstruction.ply"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"[WARNING] Data directory not found: {data_dir} — skipping track {track_num}")
        return

    print(f"\n{'#'*60}")
    print(f"# TRACK {track_num}: {data_dir}")
    print(f"{'#'*60}")

    # --- Training Steps ---

    # Step 1: Verify camera poses match the map
    run("test_camera_poses.py",
        "--data_dir", data_dir,
        "--output", vis_ply)

    # Step 2: Generate sky masks for all training images
    run("generate_sky_mask.py",
        "--data_dir", data_dir)

    # Step 3: Train 3D Gaussian Splatting model
    if not skip_train:
        train_script = "train_v2.py" if (SCRIPT_DIR / "train_v2.py").exists() else "train.py"
        run(train_script,
            "--data_dir", data_dir,
            "--output_dir", output_dir)
    else:
        print(f"\n[Skipping training for track {track_num}]")
        if not trained_ply.exists():
            print(f"[WARNING] No checkpoint found at {trained_ply}")

    # --- Evaluation Steps ---

    # Step 4: Interpolate camera poses for test frames
    run("generate_test_pose.py",
        "--data_dir", data_dir,
        "--track_dir", track_dir,
        "--track_num", track_num)

    # Step 5: Render test images from trained Gaussians
    run("generate_submission.py",
        "--ply_file", trained_ply,
        "--data_dir", data_dir,
        "--track_dir", track_dir,
        "--track_num", track_num)

    # Step 6: Zip rendered images + submission.csv
    make_zip(track_dir, track_num)

    print(f"\n[Track {track_num} complete] submission.zip → {track_dir / 'submission.zip'}")


def main():
    parser = argparse.ArgumentParser(description="3DGS SLAM automated pipeline")
    parser.add_argument("--tracks", nargs="+", type=int, default=[1, 2, 3],
                        help="Track numbers to process (default: 1 2 3)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training and go straight to evaluation")
    args = parser.parse_args()

    for track in args.tracks:
        run_track(track, skip_train=args.skip_train)

    print(f"\n{'#'*60}")
    print("All tracks complete!")
    for track in args.tracks:
        zip_path = SCRIPT_DIR / f"track{track}" / "submission.zip"
        status = "OK" if zip_path.exists() else "MISSING"
        print(f"  track{track}/submission.zip  [{status}]")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
