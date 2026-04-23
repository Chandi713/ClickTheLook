import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import LOCAL_PATHS, YOLO_DATASET_DIR


def get_disk_free_gb(path: str) -> float:
    stat = os.statvfs(path)
    return (stat.f_bavail * stat.f_frsize) / (1024 ** 3)


def verify_local_paths():
    print("Verifying data paths...")
    print("=" * 60)
    ok = True
    for name, path in LOCAL_PATHS.items():
        exists = os.path.exists(path)
        print(f"  {'OK' if exists else 'MISSING'} {name}: {path}")
        if not exists:
            ok = False
    print("=" * 60)
    return ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify data paths and check disk free space.")
    parser.add_argument("--path", default=None, help="Path to check disk space (default: YOLO_DATASET_DIR).")
    args = parser.parse_args()

    verify_local_paths()
    check_path = args.path or YOLO_DATASET_DIR
    print(f"\nDisk free at {check_path}: {get_disk_free_gb(check_path):.1f} GB")
