import cv2
import numpy as np


# Преобразование в HSV и создание маски
def create_mask(image, color_to_detect):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if color_to_detect == 1:  # Желтый
        lower_color = np.array([20, 100, 100])
        upper_color = np.array([30, 255, 255])
    elif color_to_detect == 2:  # Красный
        lower_color = np.array([0, 100, 100])
        upper_color = np.array([10, 255, 255])
    elif color_to_detect == 3:  # Синий
        lower_color = np.array([100, 100, 100])
        upper_color = np.array([130, 255, 255])

    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    selection = cv2.bitwise_and(image, image, mask=mask)
    return mask, selection


# Обработка изображения (размытие, Кэнни, морфология)
def process_image(mask):
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    canny = cv2.Canny(blurred, 10, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    return closed


# Поиск контуров
def find_contours(processed_image):
    contours = cv2.findContours(processed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours


# Рисование контуров на изображении
def draw_contours(image, contours, shape_to_detect):
    image_with_contours = image.copy()

    for cont in contours:
        sm = cv2.arcLength(cont, True)
        apd = cv2.approxPolyDP(cont, 0.04 * sm, True)

        if shape_to_detect == 0:
            cv2.drawContours(image_with_contours, [apd], -1, (255, 0, 255), 2)
        else:
            if shape_to_detect == 1 and len(apd) == 3:  # Треугольник
                cv2.drawContours(image_with_contours, [cont], -1, (255, 0, 255), 3)
            elif shape_to_detect == 2 and len(apd) == 4:  # Квадрат
                cv2.drawContours(image_with_contours, [cont], -1, (255, 0, 255), 3)
            elif shape_to_detect == 3 and len(apd) == 5:  # Пятиугольник
                cv2.drawContours(image_with_contours, [cont], -1, (255, 0, 255), 3)
            elif shape_to_detect == 4 and 6 <= len(apd) <= 12:  # Звезда
                cv2.drawContours(image_with_contours, [cont], -1, (255, 0, 255), 3)

    return image_with_contours
