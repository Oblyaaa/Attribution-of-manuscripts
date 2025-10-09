# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки новых улучшений в системе атрибуции рукописей
"""

import os
import sys
from handwriting_attribution import HandwritingAttribution

def test_dataset_loading():
    """Тест загрузки датасета и анализа баланса классов"""
    print("=== ТЕСТ 1: Загрузка датасета и анализ баланса ===")
    
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print(f"[ERROR] Папка {data_dir} не найдена!")
        return False
    
    try:
        # Подсчитываем количество авторов
        authors = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        num_authors = len(authors)
        
        if num_authors < 2:
            print(f"[ERROR] Недостаточно авторов: {num_authors} (нужно минимум 2)")
            return False
        
        print(f"[OK] Найдено {num_authors} авторов: {', '.join(authors)}")
        
        # Инициализируем модель
        model = HandwritingAttribution(num_authors)
        
        # Загружаем датасет
        image_paths, labels = model.load_dataset(data_dir)
        
        if len(image_paths) == 0:
            print("[ERROR] Не найдено изображений!")
            return False
        
        print(f"[OK] Загружено {len(image_paths)} изображений")
        
        # Тестируем создание взвешенного семплера
        print("\n--- Тестирование взвешенного семплера ---")
        sampler = model.create_weighted_sampler(labels)
        
        if sampler is not None:
            print("[OK] Взвешенный семплер создан успешно!")
            return True
        else:
            print("[ERROR] Ошибка создания взвешенного семплера")
            return False
            
    except Exception as e:
        print(f"[ERROR] Ошибка: {e}")
        return False

def test_model_initialization():
    """Тест инициализации модели"""
    print("\n=== ТЕСТ 2: Инициализация модели ===")
    
    try:
        # Тестируем с разным количеством классов
        for num_classes in [2, 5, 8]:
            model = HandwritingAttribution(num_classes)
            print(f"[OK] Модель для {num_classes} классов инициализирована")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Ошибка инициализации модели: {e}")
        return False

def test_evaluation_methods():
    """Тест методов оценки модели"""
    print("\n=== ТЕСТ 3: Методы оценки модели ===")
    
    try:
        model = HandwritingAttribution(3)  # Тестовая модель для 3 классов
        
        # Проверяем наличие новых методов
        if hasattr(model, 'evaluate_model_detailed'):
            print("[OK] Метод evaluate_model_detailed найден")
        else:
            print("[ERROR] Метод evaluate_model_detailed не найден")
            return False
        
        if hasattr(model, 'plot_confusion_matrix'):
            print("[OK] Метод plot_confusion_matrix найден")
        else:
            print("[ERROR] Метод plot_confusion_matrix не найден")
            return False
        
        if hasattr(model, 'create_weighted_sampler'):
            print("[OK] Метод create_weighted_sampler найден")
        else:
            print("[ERROR] Метод create_weighted_sampler не найден")
            return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Ошибка тестирования методов: {e}")
        return False

def test_imports():
    """Тест импортов новых библиотек"""
    print("\n=== ТЕСТ 4: Проверка импортов ===")
    
    try:
        import seaborn as sns
        print("[OK] seaborn импортирован успешно")
        
        from sklearn.metrics import confusion_matrix, classification_report
        print("[OK] sklearn.metrics импортированы успешно")
        
        from torch.utils.data import WeightedRandomSampler
        print("[OK] WeightedRandomSampler импортирован успешно")
        
        from collections import Counter
        print("[OK] Counter импортирован успешно")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Ошибка импорта: {e}")
        print("Возможно, нужно установить недостающие зависимости:")
        print("   pip install seaborn")
        return False

def main():
    """Основная функция тестирования"""
    print("ТЕСТИРОВАНИЕ УЛУЧШЕНИЙ СИСТЕМЫ АТРИБУЦИИ РУКОПИСЕЙ")
    print("=" * 60)
    
    tests = [
        ("Импорты", test_imports),
        ("Инициализация модели", test_model_initialization),
        ("Методы оценки", test_evaluation_methods),
        ("Загрузка датасета", test_dataset_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"[OK] {test_name}: ПРОЙДЕН")
            else:
                print(f"[FAIL] {test_name}: НЕ ПРОЙДЕН")
        except Exception as e:
            print(f"[ERROR] {test_name}: ОШИБКА - {e}")
        
        print("-" * 40)
    
    print(f"\nРЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("ВСЕ УЛУЧШЕНИЯ РАБОТАЮТ КОРРЕКТНО!")
    else:
        print("Некоторые тесты не прошли. Проверьте ошибки выше.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
