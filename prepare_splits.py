"""
Chia dữ liệu thành train/valid/test theo tỉ lệ, và (tùy chọn) augment offline
để cân bằng lớp — giống cách bài báo mở rộng CCDDF từ 5.000 lên 8.000 ảnh/lớp
(tổng 16.000 ảnh cân bằng).

Đầu vào:
  <src>/real/*.jpg   <src>/fake/*.jpg
Đầu ra:
  <dst>/train|valid|test/real|fake/

Ví dụ:
  python prepare_splits.py --src raw/ccddf --dst data/ccddf --target_per_class 8000
"""

import argparse
import os
import random
import shutil

import cv2
import numpy as np

IMG_SIZE = 224
SPLITS = {"train": 0.8, "valid": 0.1, "test": 0.1}


def augment_once(img, rng):
    """Augment nhẹ: xoay <=30 độ, dịch <=20%, zoom, flip, đổi độ sáng."""
    h, w = img.shape[:2]
    angle = rng.uniform(-30, 30)
    scale = rng.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += rng.uniform(-0.2, 0.2) * w
    M[1, 2] += rng.uniform(-0.2, 0.2) * h
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    out = np.clip(out.astype(np.float32) * rng.uniform(0.9, 1.1), 0, 255)
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--target_per_class", type=int, default=0,
                    help="0 = không augment offline")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)

    for cls in ["real", "fake"]:
        files = [os.path.join(args.src, cls, f)
                 for f in sorted(os.listdir(os.path.join(args.src, cls)))]
        rng.shuffle(files)

        # augment offline để đạt target_per_class (chỉ áp cho ảnh gốc trước khi chia)
        extra = []
        if args.target_per_class > len(files):
            need = args.target_per_class - len(files)
            for i in range(need):
                extra.append(files[i % len(files)])

        n = len(files)
        n_tr = int(n * SPLITS["train"])
        n_va = int(n * SPLITS["valid"])
        parts = {"train": files[:n_tr],
                 "valid": files[n_tr:n_tr + n_va],
                 "test": files[n_tr + n_va:]}

        for split, items in parts.items():
            outdir = os.path.join(args.dst, split, cls)
            os.makedirs(outdir, exist_ok=True)
            for p in items:
                img = cv2.imread(p)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(os.path.join(outdir, os.path.basename(p)), img)
            print(f"{cls}/{split}: {len(items)} ảnh")

        # ảnh augment chỉ thêm vào train, tránh rò rỉ dữ liệu sang valid/test
        if extra:
            outdir = os.path.join(args.dst, "train", cls)
            for k, p in enumerate(extra):
                img = cv2.imread(p)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                aug = augment_once(img, nprng)
                cv2.imwrite(os.path.join(outdir, f"aug_{k:06d}.jpg"), aug)
            print(f"{cls}/train: thêm {len(extra)} ảnh augment")

    print("Hoàn tất tại", args.dst)


if __name__ == "__main__":
    main()
