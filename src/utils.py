"""
utils.py — вспомогательные функции для проекта детекции людей на видео.
"""

import cv2
from typing import Tuple


def draw_box_label(
    frame,
    box: Tuple[int, int, int, int],
    label: str,
    conf: float,
    color=(0, 255, 0),
    thickness: int = 2
) -> None:
    """
    Рисует прямоугольник и подпись на кадре видео.

    Args:
        frame: numpy-массив кадра (BGR).
        box: кортеж (x1, y1, x2, y2) — координаты рамки.
        label: текст класса (например, "person").
        conf: уверенность модели (0–1).
        color: цвет рамки в формате BGR (по умолчанию зелёный).
        thickness: толщина линии рамки.
    """
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    text = f"{label}: {conf:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    # Рисуем фон под текстом
    cv2.rectangle(
        frame,
        (x1, y1 - text_h - baseline - 4),
        (x1 + text_w + 2, y1),
        color,
        thickness=cv2.FILLED,
    )
    # Рисуем текст
    cv2.putText(
        frame,
        text,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def create_video_writer(output_path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """
    Создаёт объект для записи видеофайла.

    Args:
        output_path: путь к выходному видеофайлу.
        fps: количество кадров в секунду.
        size: размер кадра (ширина, высота).

    Returns:
        Объект cv2.VideoWriter для записи видео.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)
    return writer
