"""
Suy luận trên một ảnh hoặc một thư mục ảnh, có thể dùng ensemble nhiều model.

  python predict.py --models runs/140k/best.keras runs/celebdf/best.keras \
                    --weights 0.6 0.4 --input anh_can_kiem_tra.jpg
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from losses import binary_focal_loss

IMG_SIZE = 224
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_image(path):
    img = tf.keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    return tf.keras.utils.img_to_array(img) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--input", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    models = [tf.keras.models.load_model(
        p, custom_objects={"binary_focal_loss": binary_focal_loss()})
        for p in args.models]

    w = np.array(args.weights) if args.weights else np.ones(len(models))
    w = w / w.sum()

    if os.path.isdir(args.input):
        paths = [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
                 if f.lower().endswith(EXT)]
    else:
        paths = [args.input]

    batch = np.stack([load_image(p) for p in paths])
    probs = sum(wi * m.predict(batch, verbose=0).ravel() for wi, m in zip(w, models))

    for p, pr in zip(paths, probs):
        # nhãn: real = 1, fake = 0
        label = "REAL" if pr >= args.threshold else "FAKE"
        conf = pr if label == "REAL" else 1 - pr
        print(f"{os.path.basename(p)}: {label}  (độ tin cậy {conf*100:.2f}%, p_real={pr:.4f})")


if __name__ == "__main__":
    main()
