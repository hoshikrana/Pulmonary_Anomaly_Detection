"""
scripts/download_data.py — download Kermany dataset via Kaggle API.

Setup:
  1. kaggle.com → Account → API → Create New Token → saves kaggle.json
  2. Place kaggle.json at ~/.kaggle/kaggle.json  (or upload in Colab)
  3. Run: python scripts/download_data.py

On Colab, run this cell first:
  from google.colab import files
  files.upload()   # upload kaggle.json
  import os, shutil
  os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
  shutil.move("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
"""

import os
import subprocess
import zipfile
import sys

# Ensure the project root is in the Python path before importing config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config


def main():
    dest = os.path.join(config.BASE_DIR, "data")
    os.makedirs(dest, exist_ok=True)

    print("Downloading Kermany chest X-ray dataset from Kaggle...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "paultimothymooney/chest-xray-pneumonia",
        "-p", dest,
    ], check=True)

    zip_path = os.path.join(dest, "chest-xray-pneumonia.zip")
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    os.remove(zip_path)

    xray_path = os.path.join(dest, "chest_xray")
    print(f"\nDataset ready at: {xray_path}")
    for split in ("train", "test", "val"):
        for cls in ("NORMAL", "PNEUMONIA"):
            folder = os.path.join(xray_path, split, cls)
            n = len(os.listdir(folder)) if os.path.isdir(folder) else 0
            print(f"  {split}/{cls}: {n} images")


if __name__ == "__main__":
    main()
