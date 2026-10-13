"""
Tiền xử lý Celeb-DF v2: tách frame từ video, phát hiện + crop mặt bằng MTCNN
(Mục III.B của bài báo).

Điểm quan trọng theo bài báo:
  - Dùng MTCNN (Multi-task Cascaded CNN) để phát hiện mặt.
  - Celeb-DF v2 mất cân bằng lớp (fake nhiều hơn real) -> lấy NHIỀU frame hơn
    từ video real để cân bằng tập huấn luyện.
  - Ảnh Celeb-DF v2 độ phân giải thấp -> nới rộng bounding box (margin) để lấy
    thêm ngữ cảnh quanh mặt, vì vùng nền chứa artifact hữu ích.
  - Resize toàn bộ về 224 x 224.

Cách dùng:
  python extract_faces.py --video_dir data/CelebDF/Celeb-real     --out data/celebdf/real --frames_per_video 15
  python extract_faces.py --video_dir data/CelebDF/Celeb-synthesis --out data/celebdf/fake --frames_per_video 4
"""

import argparse
import os

import cv2
import numpy as np
from mtcnn import MTCNN

IMG_SIZE = 224


def crop_with_margin(img, box, margin=0.30):
    """Crop mặt và nới rộng box thêm `margin` (30%) để giữ ngữ cảnh."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + margin)
    x1 = int(max(cx - side / 2, 0))
    y1 = int(max(cy - side / 2, 0))
    x2 = int(min(cx + side / 2, img.shape[1]))
    y2 = int(min(cy + side / 2, img.shape[0]))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return img[y1:y2, x1:x2]


def process_video(path, detector, out_dir, frames_per_video, margin):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0

    # lấy mẫu frame rải đều toàn video
    idxs = np.linspace(0, total - 1, num=min(frames_per_video, total)).astype(int)
    base = os.path.splitext(os.path.basename(path))[0]
    saved = 0

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        if not faces:
            continue
        # lấy mặt có confidence cao nhất
        face = max(faces, key=lambda f: f["confidence"])
        if face["confidence"] < 0.90:
            continue
        crop = crop_with_margin(rgb, face["box"], margin)
        if crop is None:
            continue
        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        out = os.path.join(out_dir, f"{base}_f{i:05d}.jpg")
        cv2.imwrite(out, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved += 1

    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames_per_video", type=int, default=10,
                    help="Đặt cao hơn cho video real để cân bằng lớp")
    ap.add_argument("--margin", type=float, default=0.30)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    detector = MTCNN()

    videos = [f for f in sorted(os.listdir(args.video_dir))
              if f.lower().endswith((".mp4", ".avi", ".mov"))]
    total = 0
    for n, v in enumerate(videos, 1):
        total += process_video(os.path.join(args.video_dir, v), detector,
                               args.out, args.frames_per_video, args.margin)
        print(f"[{n}/{len(videos)}] {v} -> tổng {total} ảnh mặt", flush=True)

    print("Hoàn tất. Số ảnh đã lưu:", total)


if __name__ == "__main__":
    main()
