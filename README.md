# 🖋️ Handwriting Attribution System | Система Атрибуции Рукописей

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green.svg)
![License](https://img.shields.io/badge/License-MIT-grey.svg)

**Гибридная система идентификации автора рукописного текста на основе Deep Learning и биометрического анализа**

*Дипломный проект | ResNet-50 Backbone | Zenith Infinity Module*

</div>

---

## 📖 О проекте

Данный проект решает задачу **криминалистической атрибуции (определения авторства)** рукописных документов в условиях малой обучающей выборки. 

В отличие от стандартных классификаторов, система использует **гибридный подход**:
1.  **Deep Learning:** Сверточная нейросеть (ResNet-50) для извлечения абстрактных визуальных признаков.
2.  **Computer Vision (Zenith Infinity):** Модуль интерпретируемых метрик (анализ микро-тремора, наклона, нажима), имитирующий работу эксперта-почерковеда.

Система успешно протестирована на архивных рукописях русских классиков (Есенин, Тютчев, Толстой и др.), показав высокую устойчивость к артефактам сканирования и цвету бумаги.

## ✨ Ключевые особенности

* 🚀 **Точность >95% на тестовой выборке**: Достигнута благодаря кастомной архитектуре классификатора ("Custom Head") и стратегии валидации Stratified Hold-Out.
* 🧠 **Модуль Zenith Infinity**: Вычисляет физические параметры почерка (энтропия наклона, фрактальная размерность) для верификации нейросетевого решения.
* 🛡️ **Устойчивость к малым данным**: Реализована агрессивная аугментация «на лету» (Albumentations) и калибровка эпох (Oversampling), предотвращающие переобучение на выборках <100 изображений.
* 🔍 **Интерпретируемость (XAI)**: Встроенная поддержка **Grad-CAM** для визуализации зон внимания нейросети (почему принято решение).
* ⚡ **Inference in the Wild**: Система способна классифицировать изображения с произвольным соотношением сторон и разным освещением.

## 🏗️ Технологический стек

### Core
* **Язык:** Python 3.10
* **ML Framework:** PyTorch (Torchvision)
* **CV Library:** OpenCV (cv2), Albumentations

### Архитектура
* **Backbone:** ResNet-50 (Pre-trained on ImageNet, Fine-tuned)
* **Optimizer:** AdamW + ReduceLROnPlateau Scheduler
* **Loss Function:** CrossEntropyLoss

### Аналитика
* **Metrics:** Accuracy, F1-Score (Macro), Confusion Matrix
* **Visualization:** Matplotlib, Seaborn (Radar Charts)

## ⚙️ Установка

### Предварительные требования
* Python 3.8+
* CUDA-совместимая видеокарта (рекомендуется для обучения)

### Инструкция

1.  **Клонируйте репозиторий:**
    ```bash
    git clone [https://github.com/your-username/handwriting-attribution.git](https://github.com/your-username/handwriting-attribution.git)
    cd handwriting-attribution
    ```

2.  **Создайте виртуальное окружение:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```

## 📁 Структура данных

Для обучения поместите изображения в папку `data/train`, разбив их по папкам с именами авторов:

```text
data/
└── train/
    ├── Esenin/
    │   ├── scan_001.jpg
    │   └── ...
    ├── Tyutchev/
    │   ├── doc_12.png
    │   └── ...
    └── Tolstoy/
        └── ...
