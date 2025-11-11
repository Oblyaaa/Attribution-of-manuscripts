from handwriting_attribution import HandwritingAttribution
import os
import sys

def run_training(): # Запускает процесс обучения модели
    print("\n" + "-" * 20 + " Обучение новой модели " + "-" * 20)

    # Шаг 1: Проверка данных и получение количества авторов
    data_dir = 'data/'
    if not os.path.isdir(data_dir):
        print(f"\n[ОШИБКА] Папка '{data_dir}' для обучения не найдена. Пожалуйста, создайте ее.")
        return

    authors = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if len(authors) < 2:
        print(f"\n[ОШИБКА] В папке '{data_dir}' найдено менее 2 авторов. Обучение невозможно.")
        return

    num_authors = len(authors)
    print(f"\n[INFO] Найдено {num_authors} авторов. Начинаем подготовку к обучению...")

    # Шаг 2: Получение параметров обучения от пользователя
    try:
        epochs = int(input(f"   Введите количество эпох: "))
        batch_size = int(
            input(f"   Введите размер пакета: "))
        learning_rate = float(input(f"   Введите скорость обучения: "))
    except ValueError:
        print("\n[ОШИБКА] Неверный ввод. Пожалуйста, вводите только числа.")
        return

    # Шаг 3: Вызов функции обучения из основного класса
    print("\n[INFO] Инициализация модели и запуск обучения...")
    model = HandwritingAttribution(num_authors)
    model.train(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    print("\n[SUCCESS] Обучение успешно завершено!")
    print(f"[INFO] Модель сохранена в 'handwriting_model.pth', метки в 'labels.json'.")


def run_prediction(): # Запускает процесс предсказания на основе обученной модели
    

    print("\n" + "-" * 20 + " Предсказание по изображению " + "-" * 20)

    # Шаг 1: Проверка наличия файлов модели
    model_path = 'handwriting_modelUp.pth'
    labels_path = 'labelsUp.json'
    if not (os.path.exists(model_path) and os.path.exists(labels_path)):
        print(f"\n[ОШИБКА] Файлы '{model_path}' или '{labels_path}' не найдены.")
        print("[INFO] Сначала необходимо обучить модель, выбрав опцию '1'.")
        return

    # Шаг 2: Инициализация и загрузка модели
    try:
        # Загружаем метки, чтобы узнать кол-во классов для инициализации модели
        with open(labels_path, 'r', encoding='utf-8') as f:
            import json
            num_authors = len(json.load(f)['author_to_label'])

        print(f"\n[INFO] Загрузка модели, обученной для {num_authors} авторов...")
        # Инициализируем класс с правильным количеством авторов
        model = HandwritingAttribution(num_authors)
        # Вызываем функции загрузки из основного класса
        model.load_labels(labels_path)
        model.load_model(model_path)
        print("[OK] Модель и метки успешно загружены.")
    except Exception as e:
        print(f"\n[ОШИБКА] Не удалось загрузить модель: {e}")
        return

    # Шаг 3: Получение пути к изображению и запуск предсказания
    image_path = input("   Введите путь к изображению для анализа: ").strip().replace('"', '')
    if not os.path.exists(image_path):
        print(f"\n[ОШИБКА] Изображение по пути '{image_path}' не найдено.")
        return

    # Вызываем функцию предсказания из основного класса
    try:
        results = model.predict(image_path, top_k=8)
        print("\n   [РЕЗУЛЬТАТ] Наиболее вероятные авторы:")
        for i, result in enumerate(results, 1):
            print(f"      {i}. Автор: {result['author']:<25} | Уверенность: {result['confidence']:.1f}%")
    except Exception as e:
        print(f"\n[ОШИБКА] Произошла ошибка во время анализа: {e}")


def main():
    
    if len(sys.argv) > 1:
        # Позволяет запускать скрипт с аргументами, например: python example_usage.py train
        command = sys.argv[1]
        if command == "train":
            run_training()
        elif command == "predict":
            run_prediction()
        else:
            print(f"[ОШИБКА] Неизвестная команда: {command}. Доступные команды: 'train', 'predict'.")
        return

    # Интерактивное меню, если скрипт запущен без аргументов
    while True:
        print("\n" + "=" * 60)
        print("=== Система атрибуции рукописей: Главное меню ===")
        print("=" * 60)
        print("1. Обучить новую модель")
        print("2. Использовать обученную модель для предсказания")
        print("3. Выход")
        choice = input("Ваш выбор (1-3): ").strip()

        if choice == '1':
            run_training()
        elif choice == '2':
            run_prediction()
        elif choice == '3':
            print("\n[INFO] Завершение работы.")
            break
        else:
            print("\n[ПРЕДУПРЕЖДЕНИЕ] Неверный выбор. Введите число от 1 до 3.")

        input("\nНажмите Enter для возврата в меню")


if __name__ == "__main__":
    main()
