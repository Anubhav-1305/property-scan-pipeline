"""One command per capture.
Usage: python run.py --tier {lidar,video,photo} --input <capture_dir> --out <out_dir>
"""
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=["lidar", "video", "photo"], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    raise NotImplementedError(f"tier {args.tier} not implemented yet")

if __name__ == "__main__":
    main()
