import cv2
import glob
import matplotlib.image as mpimg
import numpy as np
import os
import pandas as pd
import torch

import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset


class dataset_facial_key (Dataset): # Создание набора данных о ключевых точках на лице

    def __init__(self, csv_file, root_dir, transform=None):

        # На вход принимает csv-файл, путь к изображениям и преобразование данных (опционально):
        self.dataframe_r = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self): # Размер датасета
        return len(self.dataframe_r)

    def __getitem__(self, idx):

        image_name = os.path.join(self.root_dir, self.dataframe_r.iloc[idx, 0])        
        image = mpimg.imread(image_name)
        
        # Удаление alpha-канала изображения (если он присутствует)
        if (image.ndim > 2):        
            if(image.shape[2] == 4):
                image = image[:,:,0:3]
        
        points = self.dataframe_r.iloc[idx, 1:].to_numpy()
        points = points.astype('float').reshape(-1, 2)
        pryklad = {'image': image, 'keypoints': points}

        if self.transform: pryklad = self.transform(pryklad)

        return pryklad    


# Преобразование цветного изображения в чёрно-белое + нормализация цветового диапазона к [0,1].
class Normalize(object):
        

    def __call__(self, pryklad):
        image, points = pryklad['image'], pryklad['keypoints']
        
        image_r = np.copy(image)
        points_r = np.copy(points)

        # Конвертация изображения в чёрно-белое
        if (image.ndim > 2): image_r = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else: image_r = image
        
        # Преобразования цветового диапазона от [0, 255] к [0, 1]
        image_r = image_r / 255.0
            
        
        # Масштабирование ключевых точек так, чтобы они были сосредоточены вокруг нуля, диапазон [-0.5, 0.5].
        points_r = points_r / 224 - 0.5


        return {'image': image_r, 'keypoints': points_r}


# Изменение масштаба изображения до заданного размера: output_size -- размер вывода.
class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, pryklad):

        image, points = pryklad['image'], pryklad['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):

            if h > w: new_h, new_w = self.output_size * h / w, self.output_size
            else: new_h, new_w = self.output_size, self.output_size * w / h

        else: new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))
     
        points = points * [new_w / w, new_h / h]
        return {'image': img, 'keypoints': points}


# Рандомное кадрированине изображения
class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, pryklad):

        image, points = pryklad['image'], pryklad['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]

        points = points - [left, top]

        return {'image': image, 'keypoints': points}

# Аугментация изображения, рандомный поворот изображения
class Augment_ON(object):

    def __call__(self, pryklad):
        image, points = pryklad['image'], pryklad['keypoints']
        
        image_r = np.copy(image)
        points_r = points.detach().clone()
        
        x_r = np.random.randint(-90, 90)
        
        image_r = TF.rotate(image, x_r)
        
        for i in range(14):
            points_r[i][0] = points[i][0] * np.cos(np.deg2rad(-x_r)) - points[i][1] * np.sin(np.deg2rad(-x_r))    
            points_r[i][1] = points[i][1] * np.cos(np.deg2rad(-x_r)) + points[i][0] * np.sin(np.deg2rad(-x_r))
        
        return {'image': image_r, 'keypoints' : points_r}


# Конвертация массива NumPy в тензоры
class ToTensor(object):
    
    def __call__(self, pryklad):

        image, points = pryklad['image'], pryklad['keypoints']
         
        # Добавим канал серого цвета, если он отсутствует (два цвета)
        if(len(image.shape) == 2): image = image.reshape(image.shape[0], image.shape[1], 1)
            
        image = image.transpose((2, 0, 1)) # Смена местами цветовых осей, так как в NumPy и Torch они немного по-другому расположены
        return {'image': torch.from_numpy(image), 'keypoints': torch.from_numpy(points)}
