"""
Утилиты для работы с нейросетью атрибуции рукописей
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_sample_dataset_structure(base_path="data"):
    """Создание структуры папок для датасета"""
    
    print(f"Создание структуры датасета в папке '{base_path}'...")
    
    # Создаем основную папку
    Path(base_path).mkdir(exist_ok=True)
    
    # Создаем папки для авторов
    authors = ["author1", "author2", "author3", "author4", "author5"]
    
    for author in authors:
        author_path = Path(base_path) / author
        author_path.mkdir(exist_ok=True)
        
        # Создаем README файл в каждой папке
        readme_path = author_path / "README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"Папка для изображений рукописей автора: {author}\n")
            f.write("Поместите сюда изображения рукописей в форматах: .jpg, .jpeg, .png\n")
            f.write("Рекомендуется минимум 50-100 изображений на автора\n")
    
    print(f"✅ Структура создана:")
    print(f"   {base_path}/")
    for author in authors:
        print(f"   ├── {author}/")
        print(f"   │   └── README.txt")
    print()
    print("📝 Теперь поместите изображения рукописей в соответствующие папки")

def validate_dataset(data_dir):
    """Проверка корректности датасета"""
    
    print(f"Проверка датасета в папке '{data_dir}'...")
    
    if not os.path.exists(data_dir):
        print(f"❌ Папка '{data_dir}' не найдена")
        return False
    
    authors = []
    total_images = 0
    
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            # Подсчитываем изображения в папке автора
            images = [f for f in os.listdir(item_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            authors.append({
                'name': item,
                'count': len(images),
                'path': item_path
            })
            total_images += len(images)
    
    if not authors:
        print("❌ Не найдено папок с авторами")
        return False
    
    print(f"📊 Статистика датасета:")
    print(f"   👥 Авторов: {len(authors)}")
    print(f"   🖼️  Всего изображений: {total_images}")
    print()
    
    print("📋 Детальная информация:")
    for author in authors:
        status = "✅" if author['count'] >= 50 else "⚠️" if author['count'] >= 20 else "❌"
        print(f"   {status} {author['name']}: {author['count']} изображений")
    
    print()
    if total_images < 100:
        print("⚠️  Рекомендуется больше данных для качественного обучения")
    else:
        print("✅ Датасет выглядит хорошо для обучения")
    
    return True

def preprocess_image_for_analysis(image_path, output_path=None):
    """Предобработка одного изображения для анализа"""
    
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    # Конвертируем в RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Улучшаем контраст
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Убираем шум
    image = cv2.medianBlur(image, 3)
    
    # Бинаризация
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Возвращаем RGB изображение
    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        print(f"Обработанное изображение сохранено: {output_path}")
    
    return processed

def visualize_preprocessing(image_path):
    """Визуализация процесса предобработки"""
    
    # Загружаем оригинальное изображение
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Предобрабатываем
    processed = preprocess_image_for_analysis(image_path)
    
    # Создаем визуализацию
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original)
    axes[0].set_title('Оригинальное изображение')
    axes[0].axis('off')
    
    axes[1].imshow(processed)
    axes[1].set_title('После предобработки')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def batch_preprocess_dataset(input_dir, output_dir):
    """Пакетная предобработка всего датасета"""
    
    print(f"Пакетная предобработка датасета...")
    print(f"Входная папка: {input_dir}")
    print(f"Выходная папка: {output_dir}")
    
    # Создаем выходную папку
    Path(output_dir).mkdir(exist_ok=True)
    
    processed_count = 0
    
    for author_folder in os.listdir(input_dir):
        author_path = os.path.join(input_dir, author_folder)
        if not os.path.isdir(author_path):
            continue
        
        # Создаем папку для автора в выходной директории
        output_author_path = os.path.join(output_dir, author_folder)
        Path(output_author_path).mkdir(exist_ok=True)
        
        # Обрабатываем все изображения автора
        for image_file in os.listdir(author_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(author_path, image_file)
                output_path = os.path.join(output_author_path, image_file)
                
                try:
                    preprocess_image_for_analysis(input_path, output_path)
                    processed_count += 1
                except Exception as e:
                    print(f"Ошибка обработки {input_path}: {e}")
    
    print(f"✅ Обработано изображений: {processed_count}")

def analyze_dataset_quality(data_dir):
    """Анализ качества датасета"""
    
    print("Анализ качества датасета...")
    
    image_sizes = []
    total_images = 0
    
    for author_folder in os.listdir(data_dir):
        author_path = os.path.join(data_dir, author_folder)
        if not os.path.isdir(author_path):
            continue
        
        for image_file in os.listdir(author_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(author_path, image_file)
                try:
                    with Image.open(image_path) as img:
                        image_sizes.append(img.size)
                        total_images += 1
                except Exception as e:
                    print(f"Ошибка чтения {image_path}: {e}")
    
    if not image_sizes:
        print("❌ Не найдено изображений для анализа")
        return
    
    # Анализ размеров
    widths = [size[0] for size in image_sizes]
    heights = [size[1] for size in image_sizes]
    
    print(f"📊 Статистика размеров изображений:")
    print(f"   🖼️  Всего изображений: {total_images}")
    print(f"   📏 Ширина: {min(widths)} - {max(widths)} (среднее: {np.mean(widths):.0f})")
    print(f"   📐 Высота: {min(heights)} - {max(heights)} (среднее: {np.mean(heights):.0f})")
    
    # Проверка минимального размера
    min_size = min(min(widths), min(heights))
    if min_size < 224:
        print(f"⚠️  Некоторые изображения меньше 224px ({min_size}px)")
        print("   Рекомендуется увеличить размер или использовать интерполяцию")
    else:
        print("✅ Все изображения достаточно большие для обучения")

def create_training_script():
    """Создание скрипта для обучения"""
    
    script_content = '''"""
Скрипт для обучения нейросети атрибуции рукописей
"""

from handwriting_attribution import HandwritingAttribution
from utils import validate_dataset

def main():
    # Проверяем датасет
    if not validate_dataset("data/"):
        print("❌ Проблемы с датасетом. Исправьте их перед обучением.")
        return
    
    # Инициализация модели
    num_authors = 3  # Измените на ваше количество авторов
    model = HandwritingAttribution(num_authors)
    
    print(f"🚀 Начинаем обучение модели для {num_authors} авторов...")
    
    # Обучение
    try:
        model.train(
            data_dir="data/",
            epochs=50,
            batch_size=16,
            learning_rate=0.001
        )
        print("✅ Обучение завершено успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open("train_model.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ Создан скрипт train_model.py для обучения модели")

if __name__ == "__main__":
    print("=== Утилиты для нейросети атрибуции рукописей ===\n")
    
    # Создаем структуру датасета
    create_sample_dataset_structure()
    
    # Создаем скрипт обучения
    create_training_script()
    
    print("\n🎉 Утилиты готовы к использованию!")
    print("📁 Создана структура папок для датасета")
    print("📝 Создан скрипт train_model.py")
    print("\n💡 Следующие шаги:")
    print("   1. Поместите изображения рукописей в папки авторов")
    print("   2. Запустите: python train_model.py")
