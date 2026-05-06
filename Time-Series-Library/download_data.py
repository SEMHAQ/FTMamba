"""Download ETT and Weather datasets for FTMamba experiments."""
import os
import urllib.request

BASE_URL = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"

DATASETS = {
    "ETT-small": {
        "files": ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"],
        "dir": "dataset/ETT-small",
    },
    "Weather": {
        "files": ["weather.csv"],
        "dir": "dataset/weather",
        "hf_subdir": "weather",
    },
}


def download_file(fname, dest, subdir="ETT-small"):
    if os.path.exists(dest):
        print(f"  [skip] {dest} already exists")
        return
    url = f"{BASE_URL}/{subdir}/{fname}"
    try:
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Done: {os.path.getsize(dest)} bytes")
    except Exception as e:
        print(f"  Failed: {e}")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    for name, info in DATASETS.items():
        dest_dir = os.path.join(base, info["dir"])
        os.makedirs(dest_dir, exist_ok=True)
        print(f"\n=== {name} ===")
        subdir = info.get("hf_subdir", name)
        for fname in info["files"]:
            dest = os.path.join(dest_dir, fname)
            download_file(fname, dest, subdir=subdir)

    print("\nAll datasets downloaded!")


if __name__ == "__main__":
    main()
