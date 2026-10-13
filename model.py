"""
Kiến trúc CNN + Multi-Head Self-Attention (MHSA) theo bài báo
"Attention-Enhanced CNN for High-Performance Deepfake Detection: A Multi-Dataset Study"
(S. Dasgupta et al., IEEE Access, vol. 13, 2025)

Thông số lấy đúng từ bài báo (Mục III.C, III.D):
  - Input 224 x 224 x 3
  - 4 block Conv tuần tự: 32 -> 64 -> 128 -> 256 filter, kernel 3x3
  - He normal init, L2 regularization 1e-4
  - BatchNorm -> ReLU -> MaxPool 2x2 -> Dropout mỗi block
  - Sau block 4: MHSA với 4 head, key_dim = 64
  - Dropout 0.3 trước attention, dropout 0.2 bên trong attention
  - Residual connection + LayerNormalization
  - GlobalAveragePooling -> Dense 128 -> Dense 64 -> Dense 1 (sigmoid)
  - Tổng tham số mục tiêu: ~696,001
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

IMG_SIZE = 224
NUM_HEADS = 4
KEY_DIM = 64
L2_COEF = 1e-4


def conv_block(x, filters, dropout_rate, name):
    """Một block convolution: Conv -> BN -> ReLU -> MaxPool -> Dropout."""
    x = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(L2_COEF),
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name=f"{name}_pool")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_drop")(x)
    return x


def build_cnn_mhsa(img_size=IMG_SIZE, num_heads=NUM_HEADS, key_dim=KEY_DIM):
    inputs = layers.Input(shape=(img_size, img_size, 3), name="input_image")

    # ----- Phần CNN: 4 block, filter tăng dần -----
    x = conv_block(inputs, 32, 0.20, "block1")    # 224 -> 112
    x = conv_block(x, 64, 0.20, "block2")         # 112 -> 56
    x = conv_block(x, 128, 0.25, "block3")        # 56  -> 28
    x = conv_block(x, 256, 0.25, "block4")        # 28  -> 14, 256 channel

    # ----- Reshape feature map thành chuỗi token cho attention -----
    # (H, W, C) -> (H*W, C): mỗi vị trí không gian là 1 "token" 256 chiều
    _, h, w, c = x.shape
    seq = layers.Reshape((h * w, c), name="to_sequence")(x)

    # Dropout 0.3 trước attention
    seq_in = layers.Dropout(0.3, name="pre_attn_dropout")(seq)

    # ----- MHSA: 4 head, key_dim 64, dropout nội bộ 0.2 -----
    attn_out, attn_scores = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=0.2,
        name="mhsa",
    )(seq_in, seq_in, return_attention_scores=True)

    # Residual connection + LayerNorm:  X' = LayerNorm(X + MHA(X,X,X))
    x = layers.Add(name="residual_add")([seq, attn_out])
    x = layers.LayerNormalization(name="layer_norm")(x)

    # ----- Aggregation + phần dense phân loại -----
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=regularizers.l2(L2_COEF), name="dense_128",
    )(x)
    x = layers.Dropout(0.3, name="dense_drop1")(x)
    x = layers.Dense(
        64, activation="relu",
        kernel_regularizer=regularizers.l2(L2_COEF), name="dense_64",
    )(x)
    x = layers.Dropout(0.2, name="dense_drop2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inputs, outputs, name="CNN_MHSA_Deepfake")
    # Lưu lại tensor attention score để dùng cho heatmap
    model.attention_scores_tensor = attn_scores
    return model


if __name__ == "__main__":
    m = build_cnn_mhsa()
    m.summary()
    print("Tổng tham số:", m.count_params())
