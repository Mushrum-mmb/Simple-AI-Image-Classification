from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torchvision.transforms import ToTensor, Normalize, Compose
import random
import numpy as np

class Datasets(Dataset):
    def __init__(self, root, is_train,height,width, size = None):
        self.images = []
        self.labels = []
        mean = np.array([0.485, 0.546, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        self.height = height
        self.width = width
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        self.categories = os.listdir(os.path.join(root, 'animals', 'test'))

        data_path = os.path.join(root, 'animals')

        if is_train:
            data_path = os.path.join(data_path, 'train')
        else:
            data_path = os.path.join(data_path, 'test')

        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path, category)
            for item in os.listdir(data_files):
                path = os.path.join(data_files, item)
                self.images.append(path)
                self.labels.append(i)

        if size is not None and size < len(self.images):
            indices = random.sample(range(len(self.images)), size)
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        path = self.images[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(self.height,self.width))
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
