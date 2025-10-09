# -*- coding: utf-8 -*-
"""
Нейросеть для атрибуции рукописей по фотографиям
Автор: AI Assistant
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import os
import json
import copy
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("[WARNING] Albumentations не установлен. Используем стандартные трансформации.")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[WARNING] TIMM не установлен. Используем только torchvision модели.")

try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False
    print("[WARNING] EfficientNet-pytorch не установлен.")

warnings.filterwarnings('ignore')

class EarlyStopping:
    """Класс для ранней остановки обучения"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print("Восстановлены лучшие веса модели")
    
    def save_checkpoint(self, model):
        """Сохранение лучших весов модели"""
        self.best_weights = copy.deepcopy(model.state_dict())

class HandwritingDataset(Dataset):
    """Датасет для рукописей"""
    
    def __init__(self, image_paths, labels, transform=None, albumentations_transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.albumentations_transform = albumentations_transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Если используем Albumentations
        if self.albumentations_transform:
            image_np = np.array(image)
            augmented = self.albumentations_transform(image=image_np)
            image = augmented['image']
        # Иначе стандартные трансформации
        elif self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

class HandwritingCNN(nn.Module):
    """CNN модель для анализа почерка с поддержкой различных архитектур"""
    
    def __init__(self, num_classes, architecture='resnet50', pretrained=True):
        super(HandwritingCNN, self).__init__()
        self.architecture = architecture
        
        if architecture == 'efficientnet-b0' and EFFICIENTNET_AVAILABLE:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0' if pretrained else None)
            num_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        elif architecture == 'mobilenetv3' and TIMM_AVAILABLE:
            self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:
            # Используем ResNet-50 как архитектуру по умолчанию
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        
    def forward(self, x):
        if self.architecture == 'mobilenetv3' and hasattr(self, 'classifier'):
            features = self.backbone(x)
            return self.classifier(features)
        else:
            return self.backbone(x)

class HandwritingAttribution:
    """Основной класс для атрибуции рукописей"""
    
    def __init__(self, num_classes, architecture='resnet50', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        self.architecture = architecture
        self.model = HandwritingCNN(num_classes, architecture).to(device)
        self.label_to_author = {}
        self.author_to_label = {}
        
    def preprocess_image(self, image_path):
        """Предобработка изображения рукописи"""
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
        
        # Бинаризация (опционально)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Возвращаем RGB изображение
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    def create_albumentations_transforms(self):
        """Создание мощных трансформаций с Albumentations"""
        if not ALBUMENTATIONS_AVAILABLE:
            return None, None
        
        train_transform = A.Compose([
            A.Resize(224, 224),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5)
            ], p=0.8),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=0.5),
                A.MedianBlur(blur_limit=5, p=0.5),
                A.GaussianBlur(blur_limit=5, p=0.5),
            ], p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.5),
                A.Sharpen(p=0.5),
                A.Emboss(p=0.5),
            ], p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return train_transform, val_transform
    
    def create_data_transforms(self):
        """Создание трансформаций для обучения (стандартные)"""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def load_dataset(self, data_dir):
        """Загрузка датасета из папки"""
        image_paths = []
        labels = []
        
        # Сканируем папки с авторами
        for author_idx, author_name in enumerate(os.listdir(data_dir)):
            author_path = os.path.join(data_dir, author_name)
            if not os.path.isdir(author_path):
                continue
                
            self.label_to_author[author_idx] = author_name
            self.author_to_label[author_name] = author_idx
            
            # Загружаем все изображения автора
            for img_file in os.listdir(author_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(author_path, img_file)
                    image_paths.append(img_path)
                    labels.append(author_idx)
        
        return image_paths, labels
    
    def create_weighted_sampler(self, labels):
        """Создание взвешенного семплера для борьбы с дисбалансом классов"""
        # Подсчитываем количество образцов для каждого класса
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        print("\n=== Анализ баланса классов ===")
        for class_idx, count in sorted(class_counts.items()):
            author_name = self.label_to_author[class_idx]
            percentage = (count / total_samples) * 100
            print(f"Автор '{author_name}': {count} образцов ({percentage:.1f}%)")
        
        # Вычисляем веса для каждого класса (обратно пропорционально частоте)
        num_classes = len(class_counts)
        class_weights = {}
        for class_idx, count in class_counts.items():
            class_weights[class_idx] = total_samples / (num_classes * count)
        
        # Создаем веса для каждого образца
        sample_weights = [class_weights[label] for label in labels]
        
        # Создаем WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        print(f"\nСоздан взвешенный семплер для балансировки {num_classes} классов")
        print("Веса классов:")
        for class_idx, weight in class_weights.items():
            author_name = self.label_to_author[class_idx]
            print(f"  {author_name}: {weight:.3f}")
        
        return sampler
    
    def train(self, data_dir, epochs=30, batch_size=4, learning_rate=0.0001, 
              patience=7, use_albumentations=True, architecture='resnet50'):
        """Обучение модели с продвинутыми техниками"""
        print(f"=== Запуск обучения с архитектурой {architecture} ===")
        
        # Пересоздаем модель с новой архитектурой если нужно
        if architecture != self.architecture:
            self.architecture = architecture
            self.model = HandwritingCNN(self.num_classes, architecture).to(self.device)
            print(f"Модель пересоздана с архитектурой: {architecture}")
        
        print("Загрузка данных...")
        image_paths, labels = self.load_dataset(data_dir)
        
        if len(image_paths) == 0:
            raise ValueError("Не найдено изображений в указанной папке")
        
        print(f"Найдено {len(image_paths)} изображений от {len(self.author_to_label)} авторов")
        
        # Разделение на train/val
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Выбор типа аугментации
        if use_albumentations and ALBUMENTATIONS_AVAILABLE:
            print("\n[INFO] Используем мощную аугментацию Albumentations")
            train_albu_transform, val_albu_transform = self.create_albumentations_transforms()
            train_dataset = HandwritingDataset(train_paths, train_labels, 
                                             albumentations_transform=train_albu_transform)
            val_dataset = HandwritingDataset(val_paths, val_labels,
                                           albumentations_transform=val_albu_transform)
        else:
            print("\n[INFO] Используем стандартную аугментацию")
            train_transform, val_transform = self.create_data_transforms()
            train_dataset = HandwritingDataset(train_paths, train_labels, train_transform)
            val_dataset = HandwritingDataset(val_paths, val_labels, val_transform)
        
        # Создание взвешенного семплера для обучающей выборки
        train_sampler = self.create_weighted_sampler(train_labels)
        
        # Создание DataLoader'ов
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Оптимизатор и функция потерь
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Используем ReduceLROnPlateau вместо StepLR
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7  # Убрали несовместимый параметр verbose=True
        )
        
        # Инициализация Early Stopping
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        # Обучение
        train_losses = []
        val_accuracies = []
        val_losses = []
        best_val_acc = 0.0
        
        print(f"\n=== Начинаем обучение на {epochs} эпох с терпением {patience} ===\n")
        
        for epoch in range(epochs):
            # Обучение
            self.model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for images, labels in train_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Валидация
            self.model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            with torch.no_grad():
                for images, labels in val_pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    acc = 100 * val_correct / val_total
                    val_pbar.set_postfix({'acc': f'{acc:.2f}%', 'loss': f'{loss.item():.4f}'})
            
            val_accuracy = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_accuracy)
            val_losses.append(avg_val_loss)

            # Получаем ТЕКУЩУЮ (старую) скорость обучения ПЕРЕД обновлением планировщика
            old_lr = optimizer.param_groups[0]['lr']

            # Обновляем планировщик. Он может изменить скорость обучения
            scheduler.step(avg_val_loss)

            # Получаем НОВУЮ скорость обучения ПОСЛЕ обновления
            # Получаем НОВУЮ скорость обучения ПОСЛЕ обновления
            new_lr = optimizer.param_groups[0]['lr']

            # Выводим основную информацию об эпохе, используя уже НОВУЮ LR
            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%, LR = {new_lr:.2e}")

            # Если скорость обучения действительно снизилась, выводим дополнительное уведомление
            if new_lr < old_lr:
                print(f"    [SCHEDULER] Скорость обучения снижена с {old_lr:.2e} до {new_lr:.2e}")
            
            # Сохраняем лучшую модель
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                self.save_model("handwriting_modelUp_best.pth")
                print(f"[BEST] Новая лучшая модель сохранена! Точность: {best_val_acc:.2f}%")
            
            # Проверяем Early Stopping
            early_stopping(avg_val_loss, self.model)
            if early_stopping.early_stop:
                print(f"\n[EARLY STOP] Ранняя остановка на эпохе {epoch+1}")
                print(f"Лучшая валидационная потеря: {early_stopping.best_loss:.4f}")
                break
        
        # Финальная оценка модели
        print("\n=== Финальная оценка модели ===\n")
        self.evaluate_model_detailed(val_loader, val_dataset)
        
        # Сохранение финальной модели
        self.save_model("handwriting_modelUp.pth")
        self.save_labels("labelsUp.json")
        
        # График обучения
        self.plot_training_history_extended(train_losses, val_accuracies, val_losses)
        
        return train_losses, val_accuracies, val_losses
    
    def predict(self, image_path, top_k=3):
        """Предсказание авторства рукописи"""
        self.model.eval()
        
        # Предобработка изображения
        processed_image = self.preprocess_image(image_path)
        
        # Трансформация для предсказания
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(processed_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for i in range(top_k):
            author = self.label_to_author[top_indices[0][i].item()]
            confidence = top_probs[0][i].item() * 100
            results.append({
                'author': author,
                'confidence': confidence
            })
        
        return results
    
    def save_model(self, path):
        """Сохранение модели"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)
        print(f"Модель сохранена в {path}")
    
    def load_model(self, path):
        """Загрузка модели"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Модель загружена из {path}")
    
    def save_labels(self, path):
        """Сохранение меток авторов"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'label_to_author': self.label_to_author,
                'author_to_label': self.author_to_label
            }, f, ensure_ascii=False, indent=2)
        print(f"Метки сохранены в {path}")
    
    def load_labels(self, path):
        """Загрузка меток авторов"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.label_to_author = {int(k): v for k, v in data['label_to_author'].items()}
            self.author_to_label = data['author_to_label']
        print(f"Метки загружены из {path}")
    
    def evaluate_model_detailed(self, val_loader, val_dataset):
        """Подробная оценка модели с classification_report и confusion matrix"""
        self.model.eval()
        
        all_predictions = []
        all_true_labels = []
        
        print("Получение предсказаний для валидационной выборки...")
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Валидация"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())

                # Получаем ВСЕ возможные метки (от 0 до N-1) и соответствующие им имена
                all_possible_labels = sorted(self.label_to_author.keys())
                author_names = [self.label_to_author[i] for i in all_possible_labels]

                # Classification Report
                print("\n=== ДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССИФИКАЦИИ ===")
                print(classification_report(
                    all_true_labels,
                    all_predictions,
                    labels=all_possible_labels,  # <-- ВОТ ИСПРАВЛЕНИЕ
                    target_names=author_names,
                    digits=3,
                    zero_division=0  # Добавляем это, чтобы избежать предупреждений для классов без примеров
                ))

                # Сохранение classification report в файл
                with open('classification_report.txt', 'w', encoding='utf-8') as f:
                    f.write("=== ДЕТАЛЬНЫЙ ОТЧЕТ ПО КЛАССИФИКАЦИИ ===\n")
                    f.write(classification_report(
                        all_true_labels,
                        all_predictions,
                        labels=all_possible_labels,  # <-- И ЗДЕСЬ ТОЖЕ
                        target_names=author_names,
                        digits=3,
                        zero_division=0  # И здесь
                    ))
        
        # Confusion Matrix
        cm = confusion_matrix(all_true_labels, all_predictions)
        self.plot_confusion_matrix(cm, author_names)
        
        # Общая точность
        overall_accuracy = accuracy_score(all_true_labels, all_predictions)
        print(f"\nОБЩАЯ ТОЧНОСТЬ МОДЕЛИ: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
        
        return all_predictions, all_true_labels
    
    def plot_confusion_matrix(self, cm, class_names):
        """Построение и сохранение матрицы ошибок"""
        plt.figure(figsize=(12, 10))
        
        # Нормализованная матрица ошибок (в процентах)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Создаем тепловую карту
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Процент предсказаний'})
        
        plt.title('Матрица ошибок (нормализованная)\nАнализ путаницы между авторами', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Предсказанный автор', fontsize=12)
        plt.ylabel('Истинный автор', fontsize=12)
        
        # Поворачиваем подписи для лучшей читаемости
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
        print("Нормализованная матрица ошибок сохранена: confusion_matrix_normalized.png")
        
        # Также сохраняем абсолютные значения
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Oranges',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Количество образцов'})
        
        plt.title('Матрица ошибок (абсолютные значения)\nКоличество правильных и неправильных предсказаний', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Предсказанный автор', fontsize=12)
        plt.ylabel('Истинный автор', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix_absolute.png', dpi=300, bbox_inches='tight')
        print("Абсолютная матрица ошибок сохранена: confusion_matrix_absolute.png")
        
        plt.show()
    
    def plot_training_history_extended(self, train_losses, val_accuracies, val_losses):
        """Расширенное построение графиков обучения"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # График потерь обучения
        axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # График точности валидации
        axes[0, 1].plot(val_accuracies, label='Validation Accuracy', color='green')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # График сравнения потерь (логарифмическая шкала)
        axes[1, 0].semilogy(train_losses, label='Train Loss', color='blue')
        axes[1, 0].semilogy(val_losses, label='Val Loss', color='red')
        axes[1, 0].set_title('Loss Comparison (Log Scale)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (log)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # График скользящего среднего точности
        if len(val_accuracies) > 5:
            window = min(5, len(val_accuracies) // 3)
            moving_avg = np.convolve(val_accuracies, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(val_accuracies, alpha=0.5, label='Raw Accuracy', color='lightgreen')
            axes[1, 1].plot(range(window-1, len(val_accuracies)), moving_avg, 
                          label=f'Moving Average ({window})', color='darkgreen', linewidth=2)
        else:
            axes[1, 1].plot(val_accuracies, label='Validation Accuracy', color='green')
        
        axes[1, 1].set_title('Validation Accuracy (Smoothed)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_extended.png', dpi=300, bbox_inches='tight')
        print("Расширенные графики обучения сохранены: training_history_extended.png")
        plt.show()
    
    def plot_training_history(self, train_losses, val_accuracies):
        """Построение графика обучения (обратная совместимость)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        
        ax2.plot(val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

def create_sample_dataset():
    """Создание примера структуры датасета"""
    sample_structure = """
    Структура папки с данными:
    data/
    ├── author1/
    │   ├── handwriting1.jpg
    │   ├── handwriting2.jpg
    │   └── ...
    ├── author2/
    │   ├── handwriting1.jpg
    │   ├── handwriting2.jpg
    │   └── ...
    └── author3/
        ├── handwriting1.jpg
        └── ...
    """
    print(sample_structure)

if __name__ == "__main__":
    # Пример использования
    print("=== Нейросеть для атрибуции рукописей ===")
    print("Создание примера структуры датасета:")
    create_sample_dataset()
    
    # Инициализация модели (замените на реальное количество авторов)
    num_authors = 8  # Количество авторов в вашем датасете
    attribution_model = HandwritingAttribution(num_authors)
    
    print(f"\nМодель инициализирована для {num_authors} авторов")
    print("Для обучения используйте: attribution_model.train('path/to/data')")
    print("Для предсказания используйте: attribution_model.predict('path/to/image.jpg')")