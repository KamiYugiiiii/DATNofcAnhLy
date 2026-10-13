"""
Huấn luyện CNN-MHSA (Mục IV.A, Table 3 của bài báo).

Siêu tham số theo bài báo:
  140k / Celeb-DF v2 : batch 64,  lr 1e-4,  50 epoch
  CCDDF (tự tạo)     : batch 128, lr 1e-5,  10 epoch
  Optimizer Adam, Focal Loss (alpha 0.25, gamma 2),
  EarlyStopping + ReduceLROnPlateau.

Ví dụ:
  python train.py --data data/140k   --name 140k    --batch 64  --lr 1e-4 --epochs 50
  python train.py --data data/celebdf --name celebdf --batch 64  --lr 1e-4 --epochs 50
  python train.py --data data/ccddf   --name ccddf   --batch 128 --lr 1e-5 --epochs 10
"""

import argparse
import json
import os

import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, CSVLogger)

from data import get_loaders
from losses import binary_focal_loss
from model import build_cnn_mhsa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Thư mục gốc dataset")
    ap.add_argument("--name", required=True, help="Tên thí nghiệm, vd: 140k")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--outdir", default="runs")
    args = ap.parse_args()

    out = os.path.join(args.outdir, args.name)
    os.makedirs(out, exist_ok=True)

    # Dùng 2 GPU như bài báo (nếu chỉ có 1 GPU thì strategy tự xử lý)
    strategy = tf.distribute.MirroredStrategy()
    print("Số thiết bị:", strategy.num_replicas_in_sync)

    train, valid, test = get_loaders(args.data, args.batch)

    with strategy.scope():
        model = build_cnn_mhsa()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=binary_focal_loss(alpha=0.25, gamma=2.0),
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
    model.summary()

    callbacks = [
        ModelCheckpoint(os.path.join(out, "best.keras"), monitor="val_auc",
                        mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                          min_lr=1e-7, verbose=1),
        CSVLogger(os.path.join(out, "history.csv")),
    ]

    model.fit(train, validation_data=valid, epochs=args.epochs,
              callbacks=callbacks, verbose=1)

    results = model.evaluate(test, return_dict=True)
    print("Kết quả trên test:", results)
    with open(os.path.join(out, "test_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
