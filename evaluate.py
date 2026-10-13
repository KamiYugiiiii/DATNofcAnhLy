"""
Đánh giá: confusion matrix, ROC/AUC, classification report
(tương ứng Figure 6-11 và Table 6 của bài báo).

  python evaluate.py --model runs/140k/best.keras --data data/140k --name 140k
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

from data import make_flow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from losses import binary_focal_loss


def load_model(path):
    return tf.keras.models.load_model(
        path, custom_objects={"binary_focal_loss": binary_focal_loss()}
    )


def predict_probs(model, flow):
    probs = model.predict(flow, verbose=1).ravel()
    return flow.classes.astype(int), probs


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["fake", "real"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion matrix ({title})")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_roc(y_true, probs, title, path):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curve ({title})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    model = load_model(args.model)
    flow = make_flow(os.path.join(args.data, args.split), args.batch,
                     ImageDataGenerator(rescale=1.0 / 255), shuffle=False)

    y_true, probs = predict_probs(model, flow)
    y_pred = (probs >= 0.5).astype(int)

    plot_confusion(y_true, y_pred, args.name,
                   os.path.join(args.outdir, f"cm_{args.name}.png"))
    auc = plot_roc(y_true, probs, args.name,
                   os.path.join(args.outdir, f"roc_{args.name}.png"))

    report = classification_report(y_true, y_pred,
                                   target_names=["fake", "real"], digits=4)
    print(report)
    summary = {
        "dataset": args.name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(auc),
        "report": report,
    }
    with open(os.path.join(args.outdir, f"metrics_{args.name}.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(args.outdir, f"probs_{args.name}.npy"), probs)
    np.save(os.path.join(args.outdir, f"labels_{args.name}.npy"), y_true)


if __name__ == "__main__":
    main()
