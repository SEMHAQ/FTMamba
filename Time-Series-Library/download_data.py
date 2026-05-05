"""Download ETT and Weather datasets for FTMamba experiments."""
import os
import urllib.request
import zipfile

DATASETS = {
    "ETT-small": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/",
        "files": ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"],
        "dir": "dataset/ETT-small",
    },
    "Weather": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/Weather/",
        "files": ["weather.csv"],
        "dir": "dataset/weather",
    },
}


def download_file(url, dest):
    if os.path.exists(dest):
        print(f"  [skip] {dest} already exists")
        return
    print(f"  Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"  Done: {os.path.getsize(dest)} bytes")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    for name, info in DATASETS.items():
        dest_dir = os.path.join(base, info["dir"])
        os.makedirs(dest_dir, exist_ok=True)
        print(f"\n=== {name} ===")
        for fname in info["files"]:
            url = info["url"] + fname
            dest = os.path.join(dest_dir, fname)
            download_file(url, dest)

    print("\nAll datasets downloaded!")


if __name__ == "__main__":
    main()
