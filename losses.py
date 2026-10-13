"""
Focal Loss cho bài toán nhị phân (Mục IV.A của bài báo).
alpha = 0.25 để xử lý mất cân bằng lớp, gamma = 2 để tập trung vào mẫu khó.

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
"""

import tensorflow as tf
from tensorflow.keras import backend as K


def binary_focal_loss(alpha=0.25, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        # p_t: xác suất mà model gán cho lớp đúng
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)

        # bce cơ sở, rồi nhân hệ số điều chỉnh (1 - p_t)^gamma
        ce = -tf.math.log(p_t)
        loss = alpha_t * tf.pow(1.0 - p_t, gamma) * ce
        return tf.reduce_mean(loss)

    loss_fn.__name__ = "binary_focal_loss"
    return loss_fn
