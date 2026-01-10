from handwriting_attribution import HandwritingAttribution
import os
import sys
import json
import subprocess
import platform
import time

# Константы имен файлов
MODEL_FILENAME = 'handwriting_modelUp_best.pth'
LABELS_FILENAME = 'labelsUp.json'


def open_file_auto(path):
    """
    Магия для преподавателя: файл открывается сам.
    """
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.call(["open", path])
        else:  # Linux
            subprocess.call(["xdg-open", path])
    except Exception as e:
        print(f"[Warning] Не удалось открыть файл автоматически: {e}")


def run_training():
    """Запускает процесс обучения модели."""
    print("\n" + "=" * 50)
    print("   МОДУЛЬ ОБУЧЕНИЯ (K-FOLD DEEP LEARNING)")
    print("=" * 50)

    data_dir = 'data/'
    if not os.path.isdir(data_dir):
        print(f"\n[ОШИБКА] Папка '{data_dir}' не найдена!")
        try:
            os.makedirs(data_dir)
            print(f"   >>> Создана пустая папка '{data_dir}'. Положите туда папки с авторами!")
        except:
            pass
        return

    # Проверка структуры
    authors = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if len(authors) < 2:
        print(f"\n[ОШИБКА] Найдено авторов: {len(authors)}. Для обучения нужно минимум 2.")
        print(f"   (Положите папки с картинками в '{data_dir}')")
        return

    print(f"\n[INFO] Обнаружены классы: {authors}")

    # Настройки для Демо-режима
    print("\n--- Параметры обучения (Enter = Авто) ---")
    try:
        k_in = input("   K-Folds (Кросс-валидация) [3]: ").strip()
        k_folds = int(k_in) if k_in else 3  # 3 фолда достаточно для демо

        ep_in = input("   Эпохи на фолд [15]: ").strip()
        epochs = int(ep_in) if ep_in else 15
    except ValueError:
        print("[!] Ошибка ввода. Используем стандартные значения.")
        k_folds, epochs = 3, 15

    # Запуск
    print(f"\n[STATUS] Запуск K-Fold валидации (k={k_folds})...")
    try:
        # Инициализируем пустой класс, он сам всё загрузит внутри train
        model = HandwritingAttribution(device='cuda' if 0 else 'cpu')
        # Если есть GPU, PyTorch сам найдет, но тут для надежности

        model.train(
            data_dir,
            k_folds=k_folds,
            epochs=epochs,
            batch_size=4,
            learning_rate=0.0001
        )
        print("\n[SUCCESS] ✔ Модель успешно обучена и сохранена.")

    except Exception as e:
        print(f"\n[CRITICAL] Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()


def run_prediction():
    """Запускает процесс экспертизы."""
    print("\n" + "=" * 50)
    print("   МОДУЛЬ ЭКСПЕРТИЗЫ (ZENITH FORENSICS)")
    print("=" * 50)

    if not (os.path.exists(MODEL_FILENAME) and os.path.exists(LABELS_FILENAME)):
        print(f"\n[ОШИБКА] Файлы модели не найдены.")
        print("   >>> Сначала выполните обучение (пункт 1).")
        return

    # 1. Загрузка конфигурации
    try:
        model = HandwritingAttribution()
        model.load_labels(LABELS_FILENAME)
        model.load_model(MODEL_FILENAME)  # Здесь модель инициализируется под нужное число классов
        print(f"[INFO] Система готова. Классов в базе: {model.num_classes}")

    except Exception as e:
        print(f"[ОШИБКА] Сбой загрузки модели: {e}")
        print("   >>> Рекомендуется переобучить модель (пункт 1).")
        return

    # 2. Цикл работы
    while True:
        print("\n" + "-" * 50)
        raw_path = input(">>> Перетащите файл рукописи сюда (или 'q' для выхода): ").strip()
        image_path = raw_path.replace('"', '').replace("'", "")  # Очистка кавычек

        if image_path.lower() in ['q', 'exit', 'quit']: break
        if not os.path.exists(image_path):
            print("[!] Файл не найден.")
            continue

        try:
            print("\n   [1/3] Нейросетевой анализ...")
            results = model.predict(image_path, top_k=3)
            print(f"\n   --- ВЕРДИКТ НЕЙРОСЕТИ ---")
            for i, res in enumerate(results, 1):
                star = "★ " if i == 1 else "  "
                print(f"   {star} {res['author']:<20} | {res['confidence']:.2f}%")

            print("\n   [2/3] Криминалистический профиль (Computer Vision)...")

            # Уникальное имя отчета
            base_name = os.path.basename(image_path).split('.')[0]
            report_name = f"REPORT_{base_name}.png"

            # ГЕНЕРАЦИЯ
            model.generate_forensic_report(image_path, save_path=report_name)

            print(f"   [ГОТОВО] ✔ Отчет: {report_name}")
            print("   [3/3] Открытие...")
            time.sleep(0.5)
            open_file_auto(report_name)

        except Exception as e:
            print(f"\n[ОШИБКА] Анализ прерван: {e}")
            import traceback
            traceback.print_exc()


def main():
    while True:
        print("\n" + "=" * 60)
        print("=== ZENITH PRIME HANDWRITING SYSTEM ===")
        print("=" * 60)
        print("1. Обучить модель (Train K-Fold)")
        print("2. Провести экспертизу (Forensic Report)")
        print("3. Выход")

        cmd = input("Ваш выбор > ").strip()
        if cmd == '1':
            run_training()
        elif cmd == '2':
            run_prediction()
        elif cmd == '3':
            break
        else:
            print("Неверная команда.")


if __name__ == "__main__":
    main()
