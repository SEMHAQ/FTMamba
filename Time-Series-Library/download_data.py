"""Download ETT and Weather datasets for FTMamba experiments."""
import os
import urllib.request
import zipfile

DATASETS = {
    "ETT-small": {
        "urls": [
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{}",
        ],
        "files": ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"],
        "dir": "dataset/ETT-small",
    },
    "Weather": {
        "urls": [
            "https://raw.githubusercontent.com/thuml/Autoformer/main/datasets/weather/weather.csv",
            "https://raw.githubusercontent.com/wuhaixu2016/ETDataset/main/Weather/weather.csv",
        ],
        "files": ["weather.csv"],
        "dir": "dataset/weather",
    },
}


def download_file(urls, fname, dest):
    if os.path.exists(dest):
        print(f"  [skip] {dest} already exists")
        return
    for url_template in urls:
        url = url_template.format(fname) if '{}' in url_template else url_template
        try:
            print(f"  Trying {url} ...")
            urllib.request.urlretrieve(url, dest)
            print(f"  Done: {os.path.getsize(dest)} bytes")
            return
        except Exception as e:
            print(f"  Failed: {e}")
    print(f"  ERROR: all URLs failed for {fname}")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    for name, info in DATASETS.items():
        dest_dir = os.path.join(base, info["dir"])
        os.makedirs(dest_dir, exist_ok=True)
        print(f"\n=== {name} ===")
        for fname in info["files"]:
            dest = os.path.join(dest_dir, fname)
            download_file(info["urls"], fname, dest)

    print("\nAll datasets downloaded!")


if __name__ == "__main__":
    main()
