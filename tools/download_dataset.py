"""
Download and prepare video datasets for skeleton extraction.

Usage:
    cd AI_Adventure_Edge
    conda activate adventure_game_jetson

    # Download HMDB51 (recommended - 2GB, has 4/5 target actions):
    python tools/download_dataset.py --dataset hmdb51

    # Download relevant UCF101 classes only (~1GB subset):
    python tools/download_dataset.py --dataset ucf101

    # Then extract skeletons:
    python tools/extract_skeletons.py --input data/hmdb51/ --output data/extracted --hmdb51
"""
import argparse
import os
import subprocess
import sys


def run(cmd: str, cwd: str | None = None) -> bool:
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    return result.returncode == 0


def download_hmdb51(data_dir: str):
    """Download and extract HMDB51 dataset."""
    hmdb_dir = os.path.join(data_dir, "hmdb51")
    rar_path = os.path.join(data_dir, "hmdb51_org.rar")

    if os.path.isdir(hmdb_dir) and len(os.listdir(hmdb_dir)) > 10:
        print(f"HMDB51 already extracted at {hmdb_dir}")
        return hmdb_dir

    url = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"

    print("=" * 60)
    print("  Downloading HMDB51 (~2 GB)")
    print("=" * 60)

    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(rar_path):
        if not run(f"wget -c --progress=bar:force '{url}' -O '{rar_path}'"):
            print("ERROR: Download failed. Try manually:")
            print(f"  wget '{url}' -O '{rar_path}'")
            return None
    else:
        print(f"  RAR already downloaded: {rar_path}")

    # Check for unrar
    if subprocess.run("which unrar", shell=True, capture_output=True).returncode != 0:
        print("Installing unrar...")
        run("sudo apt-get update && sudo apt-get install -y unrar")

    # Extract outer rar
    print("\nExtracting HMDB51...")
    os.makedirs(hmdb_dir, exist_ok=True)
    if not run(f"unrar x -o- '{rar_path}' '{hmdb_dir}/'"):
        # Try with p7zip as fallback
        print("Trying p7zip fallback...")
        if subprocess.run("which 7z", shell=True, capture_output=True).returncode != 0:
            run("sudo apt-get install -y p7zip-full")
        run(f"7z x '{rar_path}' -o'{hmdb_dir}/'")

    # HMDB51 has inner .rar files per class
    inner_rars = [f for f in os.listdir(hmdb_dir) if f.endswith(".rar")]
    if inner_rars:
        print(f"\nExtracting {len(inner_rars)} inner class archives...")
        for i, rar_name in enumerate(sorted(inner_rars)):
            class_name = rar_name.replace(".rar", "")
            class_dir = os.path.join(hmdb_dir, class_name)
            if os.path.isdir(class_dir) and os.listdir(class_dir):
                continue
            os.makedirs(class_dir, exist_ok=True)
            inner_rar = os.path.join(hmdb_dir, rar_name)
            run(f"unrar x -o- '{inner_rar}' '{class_dir}/'")
            if (i + 1) % 10 == 0:
                print(f"  Extracted {i+1}/{len(inner_rars)} classes...")

        # Clean up inner rars to save space
        print("Cleaning up inner .rar files...")
        for rar_name in inner_rars:
            os.remove(os.path.join(hmdb_dir, rar_name))

    # Show relevant classes
    relevant = ["stand", "jump", "push", "run", "walk", "sit", "situp", "punch", "kick", "climb_stairs"]
    print(f"\n{'=' * 60}")
    print("  HMDB51 relevant classes:")
    for cls in relevant:
        cls_dir = os.path.join(hmdb_dir, cls)
        if os.path.isdir(cls_dir):
            count = len([f for f in os.listdir(cls_dir) if not f.startswith(".")])
            print(f"  {cls:20s}: {count} videos")
        else:
            print(f"  {cls:20s}: NOT FOUND")
    print(f"{'=' * 60}")
    print(f"\nNext: python tools/extract_skeletons.py -i {hmdb_dir} -o data/extracted --hmdb51")

    return hmdb_dir


def download_ucf101(data_dir: str):
    """Download UCF101 dataset (or guide user to download relevant subset)."""
    ucf_dir = os.path.join(data_dir, "UCF101")
    rar_path = os.path.join(data_dir, "UCF101.rar")

    if os.path.isdir(ucf_dir) and len(os.listdir(ucf_dir)) > 10:
        print(f"UCF101 already extracted at {ucf_dir}")
        return ucf_dir

    url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"

    print("=" * 60)
    print("  Downloading UCF101 (~6.5 GB)")
    print("  This is large! Only needed for extra jump/crouch/run data.")
    print("=" * 60)

    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(rar_path):
        if not run(f"wget -c --progress=bar:force '{url}' -O '{rar_path}'"):
            print("ERROR: Download failed.")
            return None
    else:
        print(f"  RAR already downloaded: {rar_path}")

    if subprocess.run("which unrar", shell=True, capture_output=True).returncode != 0:
        run("sudo apt-get update && sudo apt-get install -y unrar")

    print("\nExtracting UCF101...")
    if not run(f"unrar x -o- '{rar_path}' '{data_dir}/'"):
        run("sudo apt-get install -y p7zip-full")
        run(f"7z x '{rar_path}' -o'{data_dir}/'")

    relevant = ["JumpingJack", "JumpRope", "BodyWeightSquats", "Lunges",
                 "PushUps", "BoxingPunchingBag", "Running", "WalkingWithDog"]
    print(f"\n{'=' * 60}")
    print("  UCF101 relevant classes:")
    for cls in relevant:
        cls_dir = os.path.join(ucf_dir, cls)
        if os.path.isdir(cls_dir):
            count = len([f for f in os.listdir(cls_dir) if not f.startswith(".")])
            print(f"  {cls:25s}: {count} videos")
        else:
            print(f"  {cls:25s}: NOT FOUND")
    print(f"{'=' * 60}")
    print(f"\nNext: python tools/extract_skeletons.py -i {ucf_dir} -o data/extracted --ucf101")

    return ucf_dir


def main():
    parser = argparse.ArgumentParser(description="Download video datasets for skeleton extraction")
    parser.add_argument("--dataset", choices=["hmdb51", "ucf101", "both"], default="hmdb51")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()

    if args.dataset in ("hmdb51", "both"):
        download_hmdb51(args.data_dir)

    if args.dataset in ("ucf101", "both"):
        download_ucf101(args.data_dir)


if __name__ == "__main__":
    main()
