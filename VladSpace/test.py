import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchsummary import summary

from tqdm import tqdm
import time

categories = pd.read_csv('./data/activity_categories.csv')

id_to_category = dict(zip(categories['id'], categories['category']))

class HumanPoseDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        """
        img_dir: Папка с изображениями (В моем случае, 'drive/MyDrive/DataSets/human_poses_data/img_train').
        csv_file: Путь к таблице с метками (например, 'train_answers.csv').
        transform: Трансформации для предобработки изображений.
        """
        self.img_dir = img_dir
        self.labels = pd.read_csv(csv_file)  # Загружаем таблицу меток
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Достаем имя изображения и метку
        img_id = self.labels.iloc[idx, 0]  # img_id (имя изображения)
        label = self.labels.iloc[idx, 1]  # target_feature (метка)

        # Загружаем изображение
        img_path = os.path.join(self.img_dir, str(img_id)+'.jpg')
        image = Image.open(img_path).convert("RGB")  # Убедимся, что изображение в RGB

        # Применяем трансформации
        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((227, 227)),  # Изменяем размер изображений
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    print("Создание DataLoader")

    dataset = HumanPoseDataset(
        img_dir='./data/img_train',
        csv_file='./data/train_answers.csv',
        transform=transform
    )
    train_dataset, _ = random_split(dataset, [int(0.9*len(dataset)), len(dataset) - int(0.9*len(dataset))])

    for num_workers in [8, 10]:
        dataloader = DataLoader(train_dataset, batch_size=64, num_workers=num_workers)

        start_time = time.time()
        for batch in tqdm(dataloader, desc=f"num_workers={num_workers}"):
            pass  # Симуляция обработки батча
        end_time = time.time()

        print(f"\nnum_workers={num_workers}, Время загрузки: {end_time - start_time:.2f} секунд")
