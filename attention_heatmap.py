"""
Trực quan hóa heatmap attention (Mục III.E, Figure 12-14 của bài báo).

Quy trình đúng theo bài báo:
  1. Tạo "intermediate model" xuất ra output của lớp conv cuối và attention score
     của lớp multi-head attention.
  2. Trung bình attention weight trên TẤT CẢ các head -> bản đồ attention không gian.
  3. Chuẩn hóa về [0, 1].
  4. Resize về kích thước ảnh gốc bằng nội suy bilinear.
  5. Overlay lên ảnh gốc với colormap và độ trong suốt có kiểm soát.
  6. Áp dụng cho cả ảnh phân loại ĐÚNG và SAI để phân tích lỗi.

  python attention_heatmap.py --model runs/140k/best.keras --data data/140k/test --name 140k
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from losses import binary_focal_loss

IMG_SIZE = 224


def build_intermediate(model):
    """Model phụ xuất ra: (feature map conv cuối, attention scores, dự đoán)."""
    conv_out = model.get_layer("block4_drop").output
    mhsa = model.get_layer("mhsa")
    seq_in = model.get_layer("pre_attn_dropout").output
    _, scores = mhsa(seq_in, seq_in, return_attention_scores=True)
    return tf.keras.Model(model.input, [conv_out, scores, model.output])


def attention_map_from_scores(scores, grid_h, grid_w):
    """
    scores shape: (batch, num_heads, seq_len, seq_len)
    Trung bình trên các head, rồi trung bình theo trục query -> mức độ được
    chú ý của từng vị trí không gian. Sau đó reshape về lưới (grid_h, grid_w).
    """
    avg_heads = tf.reduce_mean(scores, axis=1)        # (b, seq, seq)
    per_token = tf.reduce_mean(avg_heads, axis=1)     # (b, seq)
    amap = tf.reshape(per_token, (-1, grid_h, grid_w, 1))
    # chuẩn hóa từng ảnh về [0, 1]
    mn = tf.reduce_min(amap, axis=[1, 2, 3], keepdims=True)
    mx = tf.reduce_max(amap, axis=[1, 2, 3], keepdims=True)
    amap = (amap - mn) / (mx - mn + 1e-8)
    # nội suy bilinear về kích thước ảnh gốc
    return tf.image.resize(amap, (IMG_SIZE, IMG_SIZE), method="bilinear").numpy()


def overlay(img, heat, alpha=0.45):
    cmap = plt.get_cmap("jet")
    colored = cmap(heat[..., 0])[..., :3]
    return np.clip((1 - alpha) * img + alpha * colored, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="Thư mục test có 2 lớp fake/real")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n_per_case", type=int, default=4)
    ap.add_argument("--outdir", default="reports/heatmaps")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    model = tf.keras.models.load_model(
        args.model, custom_objects={"binary_focal_loss": binary_focal_loss()})
    inter = build_intermediate(model)

    ds = tf.keras.utils.image_dataset_from_directory(
        args.data, labels="inferred", class_names=["fake", "real"],
        label_mode="binary", image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32, shuffle=True)

    # 4 trường hợp như bài báo: (True, Pred)
    buckets = {(0, 0): [], (1, 1): [], (0, 1): [], (1, 0): []}

    for batch_img, batch_lbl in ds:
        x = batch_img / 255.0
        conv, scores, preds = inter(x, training=False)
        grid = int(np.sqrt(scores.shape[-1]))
        heats = attention_map_from_scores(scores, grid, grid)
        p = (preds.numpy().ravel() >= 0.5).astype(int)
        y = batch_lbl.numpy().ravel().astype(int)

        for i in range(len(y)):
            key = (int(y[i]), int(p[i]))
            if len(buckets[key]) < args.n_per_case:
                buckets[key].append((x[i].numpy(), heats[i]))
        if all(len(v) >= args.n_per_case for v in buckets.values()):
            break

    titles = {
        (0, 0): "Fake đúng (True 0, Pred 0)",
        (1, 1): "Real đúng (True 1, Pred 1)",
        (0, 1): "Fake bị nhận sai thành Real (True 0, Pred 1)",
        (1, 0): "Real bị nhận sai thành Fake (True 1, Pred 0)",
    }

    for key, items in buckets.items():
        if not items:
            continue
        n = len(items)
        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.4), squeeze=False)
        for j, (img, heat) in enumerate(items):
            axes[0][j].imshow(img)
            axes[0][j].set_title("Ảnh gốc", fontsize=9)
            axes[0][j].axis("off")
            axes[1][j].imshow(overlay(img, heat))
            axes[1][j].set_title("Attention heatmap", fontsize=9)
            axes[1][j].axis("off")
        fig.suptitle(f"{titles[key]} - {args.name}")
        fig.tight_layout()
        tag = f"{key[0]}{key[1]}"
        fig.savefig(os.path.join(args.outdir, f"heatmap_{args.name}_{tag}.png"), dpi=150)
        plt.close(fig)

    print("Đã lưu heatmap vào", args.outdir)


if __name__ == "__main__":
    main()
