"""
In this script, we will load and read the images to transform them for our model, allowing it to understand the images.
We will inherit the dataset structure from torch.utils.data.
First import:
- from torch.utils.data import Dataset: For creating custom datasets in PyTorch.
- import os: For file and directory operations.
- import numpy as np: For numerical operations and array handling.
- from torchvision.transforms import ToTensor, Normalize, Compose: For image preprocessing (converting to tensors, normalization, and chaining transformations). 
- import random: For generating random numbers and performing random operations.
- import cv2: For computer vision and image processing tasks.
"""
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision.transforms import ToTensor, Normalize, Compose
import random
import cv2
# Create a class for our dataset.
class AnimalDatasets(Dataset):
    def __init__(self, root, is_train, height, width, size= None):
        # Initialize the images and labels lists, the height and width of the images, and the categories by joining the paths of the train or validation dataset.
        self.images = []
        self.labels = []
        self.height = height
        self.width = width
        self.categories = os.listdir(os.path.join(root, 'train'))
        # Initialize the mean and standard deviation for normalization. Compose the ToTensor() and Normalize() functions to initialize the transforms.
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        self.transform = Compose([
            ToTensor(),
            Normalize(mean, std)
        ])
        # If is_train is true, we join the path to the training dataset; otherwise, we join it to the validation dataset.
        if is_train:
            data_path = os.path.join(root, 'train')
        else:
            data_path = os.path.join(root, 'val')
        # Collect all images path and label iterate to our list through loop
        for i, category in enumerate(self.categories):
            image_paths = os.path.join(data_path, category)
            for path in os.listdir(image_paths):
                image_path = os.path.join(image_paths, path)
                self.images.append(image_path)
                self.labels.append(i)
        # Collect all image paths and labels, and iterate through the loop to add them to our lists.
        if size is not None and size < len(self.images):
            indices = random.sample(range(len(self.images)), size)
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
    #Define a __len__ function to retrieve length of images.
    def __len__(self):
        return len(self.images)
    # Define a __getitem__ function to retrieve images.
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.height, self.width))
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
if __name__ == '__main__':
    """
    We can test the datasets by retrieving an image or visualizing it using plt or cv2.
    Set the dataset_path and use __getitem__ to retrieve the image and label.
    """
    dataset_path = "/content/drive/MyDrive/AnimalDataset"
    test_dataset = AnimalDatasets(dataset_path, True, 240, 240, None)
    # Retrieve an image and its label
    index = 1
    image, label = test_dataset.__getitem__(index) # Change index to retrieve different images
      
    # Display the image using matplotlib
    import matplotlib.pyplot as plt
    # Convert the tensor back to numpy for visualization
    image = image.numpy().transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean  # Denormalize
    # Plot the image
    plt.imshow(image)
    plt.title(test_dataset.categories[label])
    plt.axis('off')
    plt.show()
