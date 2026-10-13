"""
Data pipeline + augmentation đúng như bài báo (Mục IV.A).

Augmentation: rescale 1/255, rotation 30 độ, shift ngang/dọc 20%, shear,
zoom, horizontal flip, brightness nhẹ, fill_mode = 'nearest'.

Cấu trúc thư mục mong đợi:
  <dataset_root>/
    train/real/*.jpg   train/fake/*.jpg
    valid/real/*.jpg   valid/fake/*.jpg
    test/real/*.jpg    test/fake/*.jpg

Nhãn: fake = 0, real = 1 (khớp cách đọc confusion matrix trong bài báo).
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
CLASSES = ["fake", "real"]   # fake -> 0, real -> 1


def _train_gen():
    return ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.9, 1.1),
        fill_mode="nearest",
    )


def _eval_gen():
    # tập valid/test chỉ rescale, không augment
    return ImageDataGenerator(rescale=1.0 / 255)


def make_flow(directory, batch_size, subset_gen, shuffle):
    return subset_gen.flow_from_directory(
        directory,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        classes=CLASSES,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=shuffle,
    )


def get_loaders(root, batch_size):
    train = make_flow(os.path.join(root, "train"), batch_size, _train_gen(), True)
    valid = make_flow(os.path.join(root, "valid"), batch_size, _eval_gen(), False)
    test = make_flow(os.path.join(root, "test"), batch_size, _eval_gen(), False)
    return train, valid, test
