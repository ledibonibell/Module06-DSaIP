import cv2
import numpy as np
import random
import math


# Функции для проверки пересечения и генерации позиции
def check_overlap(new_shape, shapes):
    for shape in shapes:
        if np.linalg.norm(np.array(new_shape[0]) - np.array(shape[0])) < (new_shape[1] + shape[1]):
            return True
    return False


def generate_random_position(xlim, ylim, size, shapes):
    while True:
        x = random.uniform(*xlim)
        y = random.uniform(*ylim)
        if not check_overlap(((x, y), size), shapes):
            return x, y


# Функции для рисования фигур
def draw_star(image, position, size, color):
    points = []
    for i in range(10):
        angle = i * (2 * np.pi / 10) - np.pi / 2
        radius = size if i % 2 == 0 else size / 2
        x = int(position[0] + radius * np.cos(angle))
        y = int(position[1] + radius * np.sin(angle))
        points.append((x, y))
    points = np.array(points, np.int32)
    cv2.fillPoly(image, [points], color)


def draw_rounded_square(image, position, size, color):
    center_x, center_y = position
    radius = size / 2
    corner_radius = size / 5
    points = []
    for i in range(4):
        angle = i * math.pi / 2 + math.pi / 4
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        for j in range(2):
            corner_angle = angle + (j - 0.5) * math.pi / 4
            corner_x = x + corner_radius * math.cos(corner_angle)
            corner_y = y + corner_radius * math.sin(corner_angle)
            points.append((int(corner_x), int(corner_y)))
    points.append(points[0])
    cv2.drawContours(image, [np.array(points)], 0, color, -1)


def draw_rounded_triangle(image, position, size, color):
    center_x, center_y = position
    radius = size / 2
    corner_radius = size / 5
    points = []
    for i in range(3):
        angle = i * 2 * math.pi / 3 + math.pi / 4
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        for j in range(2):
            corner_angle = angle + (j - 0.5) * math.pi / 6
            corner_x = x + corner_radius * math.cos(corner_angle)
            corner_y = y + corner_radius * math.sin(corner_angle)
            points.append((int(corner_x), int(corner_y)))
    points.append(points[0])
    cv2.drawContours(image, [np.array(points)], 0, color, -1)


def draw_rounded_pentagon(image, position, size, color):
    center_x, center_y = position
    radius = size / 2
    corner_radius = size / 5
    points = []
    for i in range(5):
        angle = i * 2 * math.pi / 5 + math.pi / 4
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        for j in range(2):
            corner_angle = angle + (j - 0.5) * math.pi / 10
            corner_x = x + corner_radius * math.cos(corner_angle)
            corner_y = y + corner_radius * math.sin(corner_angle)
            points.append((int(corner_x), int(corner_y)))
    points.append(points[0])
    cv2.drawContours(image, [np.array(points)], 0, color, -1)


# Генерация цвета
def generate_color(base_color):
    if base_color == 'yellow':
        return 0, 255, 255  # Желтый
    elif base_color == 'red':
        return 0, 0, 255  # Красный
    elif base_color == 'blue':
        return 255, 0, 0  # Синий


# Генерация изображения с фигурами
def generate_image(image_height, image_width, colors, shapes):
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    shapes_positions = []

    for color in colors:
        for shape in shapes:
            num_shapes = random.randint(1, 3)
            for _ in range(num_shapes):
                size = random.randint(50, 80)
                position = generate_random_position(
                    (size, image_width - size),
                    (size, image_height - size),
                    size,
                    shapes_positions
                )
                shapes_positions.append((position, size))
                shape(image, position, size, generate_color(color))

    return image
