"""
Thí nghiệm cross-dataset (Table 4) và 2 chiến lược ensemble (Table 5) của bài báo.

  - Cross-dataset: train trên dataset A, test trên dataset B.
  - Equal Weights Ensemble: 3 model, mỗi model trọng số 1/3.
  - Dataset-Weighted Ensemble: model được train trên cùng dataset với tập test
    nhận trọng số 0.6; hai model còn lại mỗi model 0.2.

Ví dụ:
  python cross_dataset_ensemble.py \
      --models 140k=runs/140k/best.keras celebdf=runs/celebdf/best.keras ccddf=runs/ccddf/best.keras \
      --datasets 140k=data/140k celebdf=data/celebdf ccddf=data/ccddf
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data import make_flow
from losses import binary_focal_loss

SAME_W = 0.6
OTHER_W = 0.2


def parse_kv(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k] = v
    return out


def metrics(y, probs):
    pred = (probs >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y, pred) * 100, 2),
        "auc": round(roc_auc_score(y, probs) * 100, 2),
        "precision": round(precision_score(y, pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y, pred, zero_division=0) * 100, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    model_paths = parse_kv(args.models)
    data_paths = parse_kv(args.datasets)

    models = {
        k: tf.keras.models.load_model(
            v, custom_objects={"binary_focal_loss": binary_focal_loss()})
        for k, v in model_paths.items()
    }

    # ---- Lấy dự đoán của mọi model trên mọi test set ----
    probs = {}   # probs[(model, dataset)] = mảng xác suất
    labels = {}  # labels[dataset]
    for ds_name, ds_path in data_paths.items():
        flow = make_flow(os.path.join(ds_path, "test"), args.batch,
                         ImageDataGenerator(rescale=1.0 / 255), shuffle=False)
        labels[ds_name] = flow.classes.astype(int)
        for m_name, m in models.items():
            flow.reset()
            probs[(m_name, ds_name)] = m.predict(flow, verbose=1).ravel()
            print(f"Xong: model {m_name} trên test set {ds_name}")

    # ---- Table 4: ma trận cross-dataset ----
    rows = []
    for m_name, ds_name in itertools.product(models, data_paths):
        r = {"train_on": m_name, "test_on": ds_name}
        r.update(metrics(labels[ds_name], probs[(m_name, ds_name)]))
        rows.append(r)
    cross = pd.DataFrame(rows)
    cross.to_csv(os.path.join(args.outdir, "table4_cross_dataset.csv"), index=False)
    print("\n=== Table 4: Cross-dataset ===\n", cross.to_string(index=False))

    # ---- Table 5: hai kiểu ensemble ----
    ens_rows = []
    names = list(models)
    for ds_name in data_paths:
        stack = np.stack([probs[(m, ds_name)] for m in names])

        equal = stack.mean(axis=0)
        r = {"test_on": ds_name, "ensemble": "Equal Weights (1/3 each)"}
        r.update(metrics(labels[ds_name], equal))
        ens_rows.append(r)

        w = np.array([SAME_W if m == ds_name else OTHER_W for m in names])
        w = w / w.sum()
        weighted = (stack * w[:, None]).sum(axis=0)
        r = {"test_on": ds_name,
             "ensemble": f"Dataset-Weighted ({SAME_W}/{OTHER_W})"}
        r.update(metrics(labels[ds_name], weighted))
        ens_rows.append(r)

    ens = pd.DataFrame(ens_rows)
    ens.to_csv(os.path.join(args.outdir, "table5_ensemble.csv"), index=False)
    print("\n=== Table 5: Ensemble ===\n", ens.to_string(index=False))

    with open(os.path.join(args.outdir, "cross_ensemble.json"), "w") as f:
        json.dump({"cross": rows, "ensemble": ens_rows}, f, indent=2)


if __name__ == "__main__":
    main()
