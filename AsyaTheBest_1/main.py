import cv2
import os
from generation import generate_image, draw_star, draw_rounded_square, draw_rounded_triangle, draw_rounded_pentagon
from processing import create_mask, process_image, find_contours, draw_contours

# Создание папки Source
if not os.path.exists('Source'):
    os.makedirs('Source')

# Настройки
image_width = 1920
image_height = 1080
colors = ['yellow', 'red', 'blue']
shapes = [draw_star, draw_rounded_square, draw_rounded_triangle, draw_rounded_pentagon]

# Генерация изображения
image = generate_image(image_height, image_width, colors, shapes)
cv2.imwrite('Source/1_original_image.png', image)

# Настройки для поиска фигур
shape_to_detect = 0  # 0 - тест, 1 - треугольник, 2 - квадрат, 3 - пятиугольник, 4 - звезда
color_to_detect = 1  # 1 - желтый, 2 - красный, 3 - синий

# Обработка изображения
mask, selection = create_mask(image, color_to_detect)
processed_image = process_image(mask)
contours = find_contours(processed_image)

# Рисование контуров
image_with_contours = draw_contours(image, contours, shape_to_detect)

# Отображение и сохранение результатов
cv2.imshow('Original Image', image)
cv2.imshow('Mask', mask)
cv2.imshow('Image with Contours', image_with_contours)

cv2.imwrite('Source/2_mask.png', mask)
cv2.imwrite('Source/3_selection.png', selection)
cv2.imwrite('Source/4_image_with_contours.png', image_with_contours)

cv2.waitKey(0)
cv2.destroyAllWindows()
