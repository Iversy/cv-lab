import cv2
import numpy as np
from math import sqrt

# Функция для вычисления среднего и стандартного отклонения
def compute_stats(window):
    """Вычисляет среднее и стандартное отклонение для окна"""
    if window.size == 0:
        return 0, 0
    
    mean_val = np.mean(window)
    std_val = np.std(window)
    
    return mean_val, std_val

# Функция для получения внешней рамки окна
def get_outer_frame(image, x, y, window_size=11):
    """Возвращает внешнюю рамку окна размером window_size x window_size"""
    height, width = image.shape[:2]
    half_size = window_size // 2
    
    # Определяем границы окна
    x1 = max(0, x - half_size)
    y1 = max(0, y - half_size)
    x2 = min(width, x + half_size + 1)
    y2 = min(height, y + half_size + 1)
    
    # Создаем копию изображения для отрисовки рамки
    frame_img = image.copy()
    
    # Рисуем внешнюю рамку (красный цвет)
    cv2.rectangle(frame_img, (x1, y1), (x2-1, y2-1), (0, 0, 255), 1)
    
    return frame_img

# Функция для обработки событий мыши
def mouse_callback(event, x, y, flags, param):
    """Обработчик событий мыши"""
    global img, original_img, window_size
    
    if event == cv2.EVENT_MOUSEMOVE:
        height, width = img.shape[:2]
        
        # Проверяем, что координаты в пределах изображения
        if 0 <= x < width and 0 <= y < height:
            # Получаем значения RGB текущего пикселя
            if len(img.shape) == 3:  # Цветное изображение
                b, g, r = img[y, x]
                intensity = (int(r) + int(g) + int(b)) / 3
            else:  # Черно-белое изображение
                r = g = b = img[y, x]
                intensity = r
            
            # Получаем окно 11x11 вокруг текущего пикселя
            half_size = window_size // 2
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(width, x + half_size + 1)
            y2 = min(height, y + half_size + 1)
            
            window = img[y1:y2, x1:x2]
            
            # Вычисляем среднее и стандартное отклонение
            if len(img.shape) == 3:
                # Для цветного изображения вычисляем для каждого канала и берем среднее
                means = []
                stds = []
                for channel in range(3):
                    channel_window = window[:, :, channel]
                    mean_val, std_val = compute_stats(channel_window)
                    means.append(mean_val)
                    stds.append(std_val)
                mean_w = np.mean(means)
                std_w = np.mean(stds)
            else:
                mean_w, std_w = compute_stats(window)
            
            # Создаем изображение с рамкой
            frame_img = get_outer_frame(original_img.copy(), x, y, window_size)
            
            # Отображаем информацию в консоли
            print("\n" + "="*50)
            print(f"Координаты: ({x}, {y})")
            print(f"RGB значения: R={r}, G={g}, B={b}")
            print(f"Интенсивность: {intensity:.2f}")
            print(f"Среднее значение в окне: {mean_w:.2f}")
            print(f"Стандартное отклонение в окне: {std_w:.2f}")
            print("="*50)
            
            # Создаем изображение для отображения текста
            info_img = np.ones((150, 400, 3), dtype=np.uint8) * 255
            
            # Добавляем текст на изображение
            cv2.putText(info_img, f"Coords: ({x}, {y})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(info_img, f"RGB: ({r}, {g}, {b})", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(info_img, f"Intensity: {intensity:.2f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(info_img, f"Mean: {mean_w:.2f}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(info_img, f"Std: {std_w:.2f}", (200, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Показываем изображения
            cv2.imshow('Image with Frame', frame_img)
            cv2.imshow('Pixel Information', info_img)

# Основная программа
def main():
    global img, original_img, window_size
    
    window_size = 11
    
    # Загружаем изображение (замените путь на свой)
    # img = cv2.imread('your_image.jpg')
    
    # Если нет изображения, создаем тестовое
    img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    
    original_img = img.copy()
    
    # Создаем окна
    cv2.namedWindow('Image with Frame')
    cv2.namedWindow('Pixel Information')
    
    # Устанавливаем обработчик мыши
    cv2.setMouseCallback('Image with Frame', mouse_callback)
    
    # Показываем начальное изображение
    cv2.imshow('Image with Frame', img)
    
    # Создаем начальное информационное изображение
    info_img = np.ones((150, 400, 3), dtype=np.uint8) * 255
    cv2.putText(info_img, "Move mouse over the image", (50, 75), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow('Pixel Information', info_img)
    
    print("Двигайте мышью над изображением для получения информации о пикселях")
    print("Нажмите 'q' для выхода")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()