from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torchvision.transforms import ToTensor, Normalize, Compose
import random
import numpy as np

#Defines a new class Datasets that inherits from Dataset
class Datasets(Dataset):
    def __init__(self, root, is_train,height,width, size = None):
        #Initializes empty lists to hold image file paths and their corresponding labels.
        self.images = []
        self.labels = []
        #Defines the mean and standard deviation for normalizing images.
        mean = np.array([0.485, 0.546, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        #Stores the image dimensions and sets up a transformation pipeline that converts images to tensors and normalizes them using the defined mean and standard deviation.
        self.height = height
        self.width = width
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        # Lists the categories (subdirectories) of images in the test directory under animals and stores them in self.categories.
        self.categories = os.listdir(os.path.join(root, 'animals', 'test'))
        #Constructs the path to either the training or testing data based on the is_train flag.
        data_path = os.path.join(root, 'animals')
        if is_train:
            data_path = os.path.join(data_path, 'train')
        else:
            data_path = os.path.join(data_path, 'test')
        #Iterates over each category, constructs the path to each image file, and appends the full path to self.images. It also appends the corresponding label (the category index) to self.labels.
        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path, category)
            for item in os.listdir(data_files):
                path = os.path.join(data_files, item)
                self.images.append(path)
                self.labels.append(i)
        #If a size is specified and is less than the total number of images, it randomly samples the specified number of images and their labels.
        if size is not None and size < len(self.images):
            indices = random.sample(range(len(self.images)), size)
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
    #Returns the total number of samples (images) in the dataset.
    def __len__(self):
        return len(self.labels)
    #Retrieves an image and its label based on the provided index (idx).
    def __getitem__(self, idx):
        path = self.images[idx]
        image = cv2.imread(path) #Reads the image from the file.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Converts the image from BGR (OpenCV format) to RGB.
        image = cv2.resize(image,(self.height,self.width)) #Resizes the image to the specified dimensions.
        image = self.transform(image) #Applies the transformations (normalization).
        label = self.labels[idx]
        return image, label
