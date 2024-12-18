import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from tqdm import tqdm 
import random
import pandas as pd
from torch.optim import Adam
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchsummary import summary

print("CUDA available:", torch.cuda.is_available())
print("CUDA device name:", torch.cuda.get_device_name(0))
print("PyTorch CUDA version:", torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.rand(3, 3).to(device)
print("Tensor device:", x.device)

categories = pd.read_csv('../data/activity_categories.csv')


id_to_category = dict(zip(categories['id'], categories['category']))

class HumanPoseDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        """
        img_dir: Папка с изображениями (img_train).
        csv_file: Путь к таблице с метками (например, 'train_answers.csv').
        transform: Трансформации для предобработки изображений.
        """
        self.img_dir = img_dir
        self.files = os.listdir(self.img_dir)
        self.labels = pd.read_csv(csv_file)  # Загружаем таблицу меток
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Достаем имя изображения и метку
        img_id = self.files[idx]
        img_id =img_id.split('.')[0]
        # img_id1 = self.labels.iloc[idx, 0]  # img_id (имя изображения)
        # label = self.labels.iloc[idx, 1]  # target_feature (метка)
        label = self.labels.loc[self.labels['img_id'] == int(img_id), 'target_feature'].values[0]

        # Загружаем изображение
        # img_path = os.path.join(self.img_dir, 'aug_'+str(img_id)+'.jpg')
        img_path = os.path.join(self.img_dir, str(img_id)+'.jpg')
        image = Image.open(img_path).convert("RGB")

        # Применяем трансформации
        if self.transform:
            image = self.transform(image)

        return image, label, img_id
    
    def __len__(self):
        return len(self.files)
    


    
transform = transforms.Compose([
    transforms.Resize((227, 227)),  # Изменяем размер изображений
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
    
val_dataset = HumanPoseDataset(
    img_dir='../data/val_dataset',
    csv_file='../data/train_answers.csv',
    transform=transform
)

train_dataset = HumanPoseDataset(
    img_dir='../data/augmented_and_train_dataset',
    csv_file='../data/augmented_train_answers.csv',
    transform=transform
)
model_res = custom_resnet18(dropout_prob=0.5).to(device)
path_res_check = './train_info/checkpoints_14_12_17_44/best_UpdatedResNet_checkpoint_22_val_accuracy=0.6505.pt'
checkpoint = torch.load(path_res_check, map_location=torch.device(device))
model_res.load_state_dict(checkpoint["model"])

model_res_m = custom_resnet18_m(dropout_prob=0.2).to(device)
path_res_m_check = './train_info/checkpoints_14_12_18_21/best_UpdatedResNet_checkpoint_13_val_accuracy=0.6448.pt'
checkpoint = torch.load(path_res_m_check, map_location=torch.device(device))
model_res_m.load_state_dict(checkpoint["model"])

model_effnb0_m = EfficientNetB0().to(device)
path__effnb0_m = './train_info/checkpoints_16_12_22_11/best_checkpoint_20_val_F1=0.4210.pt'
checkpoint = torch.load(path__effnb0_m, map_location=torch.device(device))
model_effnb0_m.load_state_dict(checkpoint["model"])

model_effnb0 = custom_efficientnet_b0(num_classes=20)
path__effnb0 = './train_info/check_effnb0/best_checkpoint_18_val_F1=0.5903.pt'
checkpoint = torch.load(path__effnb0, map_location=torch.device(device))
model_effnb0.load_state_dict(checkpoint["model"])

model_custom_effn_401 = custom_efficientnet(num_classes=20)
model_custom_effn_300 = custom_efficientnet(num_classes=20)

path_custom_effn_401 = './train_info/checkpoints_17_12_19_24/best_checkpoint_251_val_F1=0.6286.pt'
path_custom_effn_300 = './train_info/checkpoints_17_12_19_24/best_checkpoint_142_val_F1=0.6188.pt'
checkpoint = torch.load(path_custom_effn_401, map_location=torch.device(device))
model_custom_effn_401.load_state_dict(checkpoint["model"])
checkpoint = torch.load(path_custom_effn_300, map_location=torch.device(device))
model_custom_effn_300.load_state_dict(checkpoint["model"])

model_res.eval()
model_res_m.eval()
model_effnb0_m.eval()
model_effnb0.eval()
model_custom_effn_401.eval()
model_custom_effn_300.eval()

def ensemble_predict(dataloader, model_1, model_2, model_3, model_4, model_5, model_6, device='cuda'):
    model_1.to(device)
    model_2.to(device)
    model_3.to(device)
    model_4.to(device)
    model_5.to(device)
    model_6.to(device)

    predictions = []
    images = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, _ = batch  # inputs - изображения, targets - метки
            inputs = inputs.to(device)

            # Получаем вероятности от обеих моделей
            probs_1 = F.softmax(model_1(inputs), dim=1)
            probs_2 = F.softmax(model_2(inputs), dim=1)
            probs_3 = F.softmax(model_3(inputs), dim=1)
            probs_4 = F.softmax(model_4(inputs), dim=1)
            probs_5 = F.softmax(model_5(inputs), dim=1)
            probs_6 = F.softmax(model_6(inputs), dim=1)

            # Усредняем вероятности
            ensemble_probs = (probs_1 + probs_2 + probs_3 + probs_4+probs_5+probs_6) / 6

            # Выбираем класс с максимальной вероятностью
            predicted_classes = torch.argmax(ensemble_probs, dim=1)
            predictions.extend(predicted_classes.cpu().numpy())

            # Сохраняем изображения и метки для визуализации
            images.extend(inputs.cpu())
            labels.extend(targets.cpu().numpy())

    return predictions, images, labels

from numpy import sqrt


def visualize_predictions(images, labels, predictions, class_names,  num_images= 10):
    """
    Визуализирует изображения с реальными и предсказанными метками.

    Args:
        images (list): Список изображений.
        labels (list): Реальные метки.
        predictions (list): Предсказанные метки.
        class_names (dict): Словарь с именами классов, где ключ - метка, значение - имя класса.
    """
    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        plt.subplot(int(sqrt(num_images)), int(sqrt(num_images)), i + 1)
        plt.imshow(images[i].permute(1, 2, 0))  # Преобразуем изображение для отображения
        plt.title(f"Real: {class_names[labels[i]]}\nPred: {class_names[predictions[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

from sklearn.metrics import f1_score

def calculate_f1(predictions, labels):
    """
    Вычисляет F1-меру для предсказаний ансамбля.

    Args:
        predictions (list): Список предсказанных классов.
        labels (list): Список реальных меток.

    Returns:
        float: Значение F1-меры.
    """
    return f1_score(labels, predictions, average='weighted')

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
ensemble_predictions, images, labels = ensemble_predict(val_loader, model_res, model_res_m, model_effnb0_m, model_effnb0, model_custom_effn_401, model_custom_effn_300)

f1 = calculate_f1(ensemble_predictions, labels)
print(f"f1: {f1:.2%}")