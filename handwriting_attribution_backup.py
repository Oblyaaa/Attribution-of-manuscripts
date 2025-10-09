# -*- coding: utf-8 -*-
"""
Нейросеть для атрибуции рукописей по фотографиям
Автор: AI Assistant
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class HandwritingDataset(Dataset):
    """Датасет для рукописей"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

class HandwritingCNN(nn.Module):
    """CNN модель для анализа почерка"""
    
    def __init__(self, num_classes, pretrained=True):
        super(HandwritingCNN, self).__init__()
        
        # Используем предобученную ResNet как backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Заменяем последний слой
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class HandwritingAttribution:
    """Основной класс для атрибуции рукописей"""
    
    def __init__(self, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        self.model = HandwritingCNN(num_classes).to(device)
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
    
    def create_data_transforms(self):
        """Создание трансформаций для обучения"""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    
    def train(self, data_dir, epochs=50, batch_size=16, learning_rate=0.001):
        """Обучение модели"""
        print("Загрузка данных...")
        image_paths, labels = self.load_dataset(data_dir)
        
        if len(image_paths) == 0:
            raise ValueError("Не найдено изображений в указанной папке")
        
        print(f"Найдено {len(image_paths)} изображений от {len(self.author_to_label)} авторов")
        
        # Разделение на train/val
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Создание трансформаций
        train_transform, val_transform = self.create_data_transforms()
        
        # Создание датасетов
        train_dataset = HandwritingDataset(train_paths, train_labels, train_transform)
        val_dataset = HandwritingDataset(val_paths, val_labels, val_transform)
        
        # Создание DataLoader'ов
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Оптимизатор и функция потерь
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        
        # Обучение
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Обучение
            self.model.train()
            train_loss = 0.0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Валидация
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}: Loss = {avg_train_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")
            
            scheduler.step()
        
        # Сохранение модели
        self.save_model("handwriting_model.pth")
        self.save_labels("labels.json")
        
        # График обучения
        self.plot_training_history(train_losses, val_accuracies)
        
        return train_losses, val_accuracies
    
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
    
    def plot_training_history(self, train_losses, val_accuracies):
        """Построение графика обучения"""
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