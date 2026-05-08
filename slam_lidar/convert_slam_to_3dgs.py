#!/usr/bin/env python3
"""Convert SLAM outputs (mid2dataset/<Track>) into the 3DGS input layout.

Example:
    python convert_slam_to_3dgs.py --track 1
    python convert_slam_to_3dgs.py --track 2 --copy
    python convert_slam_to_3dgs.py --track 2 --input_dir mid2dataset/Track2 \
        --output_dir itri58_colored_pcd
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

# Per-track camera intrinsics and default paths (from SLAM code).
TRACK_CONFIG = {
    1: {
        "intrinsics": {
            "fx": 653.143433778113,
            "fy": 657.670567367976,
            "cx": 299.1738577337179,
            "cy": 236.60674857178367,
            "width": 640,
            "height": 480,
        },
        "input_dir": Path("data/Track1"),
        "output_dir": Path("itri58_colored_pcd_t1"),
        "pcd_name": "Track1.pcd",
    },
    2: {
        "intrinsics": {
            "fx": 1040.18078,
            "fy": 1038.55506,
            "cx": 720.04463,
            "cy": 464.33648,
            "width": 1440,
            "height": 928,
        },
        "input_dir": Path("data/Track2"),
        "output_dir": Path("itri58_colored_pcd_t2"),
        "pcd_name": "Track2.pcd",
    },
    3: {
        "intrinsics": {
            "fx": 979.71515067,
            "fy": 986.50585105,
            "cx": 448.7607866,
            "cy": 354.91012286,
            "width": 960,
            "height": 720,
        },
        "input_dir": Path("data/Track3"),
        "output_dir": Path("itri58_colored_pcd_t3"),
        "pcd_name": "Track3.pcd",
    }
}


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.unlink(missing_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", type=int, choices=(1, 2, 3), default=None,
                    help="Track number, selects intrinsics")
    ap.add_argument("--input_dir", type=Path, default=None,
                    help="SLAM track dir (default: data/Track<track>)")
    ap.add_argument("--output_dir", type=Path, default=None,
                    help="3DGS output dir (default: itri58_colored_pcd_t<track>)")
    ap.add_argument("--pcd-name", default=None,
                    help="Source PCD filename in result/ (default: Track<track>.pcd)")
    ap.add_argument("--copy", action="store_true",
                    help="Copy files instead of creating symlinks")
    args = ap.parse_args()

    cfg = TRACK_CONFIG[args.track]
    input_dir = args.input_dir if args.input_dir is not None else cfg["input_dir"]
    output_dir = args.output_dir if args.output_dir is not None else cfg["output_dir"]
    pcd_name = args.pcd_name or cfg["pcd_name"]
    intrinsics = cfg["intrinsics"]

    src_image_dir = input_dir / "data" / "image"
    src_pose_csv = input_dir / "result" / "camera_pose.csv"
    src_pcd = input_dir / "result" / pcd_name

    for path in (src_image_dir, src_pose_csv, src_pcd):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    out = output_dir
    out_image_dir = out / "itri58_image"
    out_sky_dir = out / "sky_masks"
    out.mkdir(parents=True, exist_ok=True)
    out_image_dir.mkdir(exist_ok=True)
    out_sky_dir.mkdir(exist_ok=True)

    # Build timestamp -> 16-float pose lookup.
    df = pd.read_csv(src_pose_csv)
    if df.shape[1] != 17:
        raise ValueError(f"Expected 17 columns in {src_pose_csv}, got {df.shape[1]}")
    ts_to_pose = dict(zip(df["timestamp"].astype("int64"),
                          df.iloc[:, 1:].to_numpy()))

    images = sorted(p for p in src_image_dir.iterdir() if p.suffix == ".jpg")
    matched, skipped = [], []
    for img in images:
        try:
            ts = int(img.stem)
        except ValueError:
            skipped.append(img.name)
            continue
        pose = ts_to_pose.get(ts)
        if pose is None:
            skipped.append(img.name)
            continue
        matched.append((img, pose))

    pose_path = out / "camera_gt_pose.txt"
    with open(pose_path, "w") as f:
        for _, pose in matched:
            f.write(",".join(f"{v:.10f}" for v in pose) + "\n")

    # Refresh image dir so stale entries from a prior run don't linger.
    for existing in out_image_dir.iterdir():
        existing.unlink()
    for img, _ in matched:
        link_or_copy(img, out_image_dir / img.name, args.copy)

    link_or_copy(src_pcd, out / "itri58_full_color_map.pcd", args.copy)

    with open(out / "camera_intrinsics.json", "w") as f:
        json.dump(intrinsics, f, indent=4)

    n_pose_lines = sum(1 for _ in open(pose_path))
    n_out_imgs = sum(1 for _ in out_image_dir.iterdir())
    assert n_pose_lines == n_out_imgs == len(matched), \
        f"pose/image count mismatch: poses={n_pose_lines}, images={n_out_imgs}, matched={len(matched)}"

    print(f"Track        : {args.track}")
    print(f"Source       : {input_dir}")
    print(f"Output       : {out}")
    print(f"CSV poses    : {len(ts_to_pose)}")
    print(f"Source images: {len(images)}")
    print(f"Matched      : {len(matched)}")
    print(f"Skipped      : {len(skipped)}")
    if skipped:
        preview = ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else "")
        print(f"  (no pose)  : {preview}")
    print(f"Wrote        : {pose_path}")
    print(f"Wrote        : {out / 'camera_intrinsics.json'}")
    print(f"{'Copied' if args.copy else 'Linked'}       : {out / 'itri58_full_color_map.pcd'}")
    print(f"{'Copied' if args.copy else 'Linked'}       : {out_image_dir}/ ({n_out_imgs} files)")
    print(f"Created      : {out_sky_dir}/ (empty)")


if __name__ == "__main__":
    main()
