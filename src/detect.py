"""
detect.py — основная программа для детекции людей на видео crowd.mp4.

Описание:
    Скрипт считывает видео, применяет модель YOLOv8 для обнаружения людей
    (класс 'person'), отрисовывает рамки с подписями и сохраняет результат.

Пример запуска:
    python src/detect.py --input crowd.mp4 --output crowd_out.mp4 --weights yolo11s.pt --conf 0.4
"""

import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from utils import draw_box_label, create_video_writer


def parse_args() -> argparse.Namespace:
    """Парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Детекция людей на видео")
    parser.add_argument("--input", type=str, default="crowd.mp4",
                        help="Путь к входному видеофайлу")
    parser.add_argument("--output", type=str, default="crowd_out.mp4",
                        help="Путь к выходному видеофайлу")
    parser.add_argument("--weights", type=str, default="yolo11n.pt",
                        help="Путь к весам модели YOLO11 или её названию")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Порог уверенности для детекции")
    parser.add_argument("--device", type=str, default=None,
                        help="Устройство для инференса: cpu или cuda:0")
    return parser.parse_args()


def filter_persons(results) -> List[Tuple[int, int, int, int, float]]:
    """
    Отфильтровывает только людей из результатов YOLO.

    Возвращает список кортежей (x1, y1, x2, y2, conf)
    для класса 'person'
    """
    persons = []
    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls.cpu().numpy().item())
            conf = float(box.conf.cpu().numpy().item())
            if cls == 0:  # класс 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                persons.append((x1, y1, x2, y2, conf))
    return persons


def main():
    """Загрузка видео, инференс, отрисовка и сохранение результата."""
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Видео '{args.input}' не найдено")

    # Загружаем видео
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = create_video_writer(args.output, fps, (width, height))

    # Загружаем модель
    print("Загрузка модели YOLO...")
    model = YOLO(args.weights)
    if args.device:
        model.to(args.device)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pbar = tqdm(total=total_frames, desc="Обработка кадров")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Инференс
            results = model(frame, imgsz=640, conf=args.conf, verbose=False)
            persons = filter_persons(results)

            # Отрисовка рамок
            for (x1, y1, x2, y2, conf) in persons:
                draw_box_label(frame, (x1, y1, x2, y2), "person", conf, thickness=2)

            writer.write(frame)
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        writer.release()
        print(f"\nВидео сохранено в '{args.output}'")


if __name__ == "__main__":
    main()