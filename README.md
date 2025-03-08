# Simple-AI-Image-Classification

# Work in progress ⏳

⭐ Star me on GitHub — it motivates me a lot!

🔥 Share it if you like it!!!

[![Share](https://img.shields.io/badge/share-000000?logo=x&logoColor=white)](https://x.com/intent/tweet?text=Check%20out%20this%20project%20on%20GitHub:%20https://github.com/Abblix/Oidc.Server%20%23OpenIDConnect%20%23Security%20%23Authentication)
[![Share](https://img.shields.io/badge/share-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/sharer/sharer.php?u=https://github.com/Abblix/Oidc.Server)
[![Share](https://img.shields.io/badge/share-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/Abblix/Oidc.Server)
[![Share](https://img.shields.io/badge/share-FF4500?logo=reddit&logoColor=white)](https://www.reddit.com/submit?title=Check%20out%20this%20project%20on%20GitHub:%20https://github.com/Abblix/Oidc.Server)
[![Share](https://img.shields.io/badge/share-0088CC?logo=telegram&logoColor=white)](https://t.me/share/url?url=https://github.com/Abblix/Oidc.Server&text=Check%20out%20this%20project%20on%20GitHub)

### Table of Contents
- [About](#-about)
- [Features](#-features)
- [Installation](#%EF%B8%8F-installation)
- [Usage](#%EF%B8%8F-usage)
- [How It Works](#-how-it-works)


### 🚀 About

This AI application performs image classification using deep learning. Trained on my private datasets with the ResNet-18 model, it accurately predicts various animal categories. Users can upload images and receive predictions along with confidence scores.

* Author: [Mushrum-mmb](https://github.com/Mushrum-mmb/)
* Model: [resnet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
* Framework: [gradio](https://www.gradio.app/)

![image](https://github.com/user-attachments/assets/0ca4a168-0f6c-4644-8068-ec4efb402332)
![image](https://github.com/user-attachments/assets/c8fb6d3e-8c98-4b5d-9d13-e81f5908a1e2)


### 🎓 Features
* Image Classification:
The application allows users to upload images for classification into specific animal categories.
* Pre-trained Model:
Utilizes a pre-trained ResNet-18 model fine-tuned for classifying various animals.
* Real-time Inference:
Provides real-time feedback by displaying the predicted class and its confidence percentage upon image upload.
* Device Compatibility:
Automatically uses GPU acceleration if available, ensuring faster inference times.
* Easy Deployment:
The Gradio interface can be launched with a simple command, and the share=True option allows sharing the interface with others via a public link.


### ⬇️ Installation
***Ensure that you have already installed Git and set up your Python environment.***

To run this application locally, ensure you have the following dependencies installed:
```bash
pip install torch torchvision gradio opencv-python scikit-learn matplotlib tensorboard tqdm
```

To download the [Training artifacts](https://drive.google.com/file/d/18dlmEhR9DYf4bsbU3hq0zq9g_nekY0zO/view?usp=drive_link) click on the blue underline words or the following URL: [https://drive.google.com/file/d/18dlmEhR9DYf4bsbU3hq0zq9g_nekY0zO/view?usp=drive_link](https://drive.google.com/file/d/18dlmEhR9DYf4bsbU3hq0zq9g_nekY0zO/view?usp=drive_link)

After downloading, unzip the file.

### ▶️ Usage
Open CMD and clone the repository.
```bash
git clone https://github.com/Mushrum-mmb/Simple-AI-Image-Classification.git
```
Then cd to the clone path.
```bash
cd Simple-AI-Image-Classification
```
Copy the path of the checkpoint folder that contains best.pt; it will work like this: 

![image](https://github.com/user-attachments/assets/e7706a92-eceb-4808-b7b0-08f2f5f7fede)

Then launch the application by running run.py and paste the path of the folder containing best.pt.

#For ex: python run.py --checkpoint "C:\Users\DELL\Downloads\Training artifacts\checkpoint"
```bash
python run.py --checkpoint "your-checkpoint-path"

```
Open the provided link in your browser to access the interface. Your work will look like this after everything:

![image](https://github.com/user-attachments/assets/07360da9-aae1-4797-bfef-9f2ea7aba9a4)



### 👍 How It Works

**1. Imports and Dataset Class**
```
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torchvision.transforms import ToTensor, Normalize, Compose
import random
import numpy as np
```
Imports: `Dataset` and `DataLoader` are used for managing and loading data.
`os` is used for file and directory operations.
`cv2` is for image processing.
`torchvision.transforms` is for applying transformations to images.
`random` is used for random sampling.
`numpy` is for create number arrays.

*Dataset Class Definition*

Purpose: Defines a custom dataset class that inherits from Dataset.
```
class Datasets(Dataset):
    def __init__(self, root, is_train, height, width, size=None):
```
Parameters:
* root: Base directory of the dataset.
* is_train: Boolean indicating if the dataset is for training or testing.
* height and width: Dimensions to which images will be resized.
* size: Optional parameter to limit the number of images loaded.

*Initialization Logic*
```
self.images = []
self.labels = []
mean = np.array([0.485, 0.546, 0.406])
std = np.array([0.229, 0.224, 0.225])
self.height = height
self.width = width
self.transform = Compose([ToTensor(), Normalize(mean=mean, std=std)])
```
Lists for Images and Labels: Initializes empty lists to store image paths and their corresponding labels.

Normalization: Sets mean and standard deviation for image normalization.

Transformations: Defines a transformation pipeline to convert images to tensors and normalize them.

*Loading Data*
```
self.categories = os.listdir(os.path.join(root, 'animals', 'test'))
data_path = os.path.join(root, 'animals')
```
Categories: Lists all categories (subdirectories) in the dataset.

Data Path: Sets the path to the dataset based on whether it's for training or testing.

*Image and Label Collection*
```
for i, category in enumerate(self.categories):
    data_files = os.path.join(data_path, category)
    for item in os.listdir(data_files):
        path = os.path.join(data_files, item)
        self.images.append(path)
        self.labels.append(i)
```
Loop Through Categories: Iterates through each category and appends the image paths and corresponding labels.

*Optional Size Limiting*
```
if size is not None and size < len(self.images):
    indices = random.sample(range(len(self.images)), size)
    self.images = [self.images[i] for i in indices]
    self.labels = [self.labels[i] for i in indices]
```
Random Sampling: If a size limit is specified, it randomly samples that many images.

**2. Length and Get Item Methods**
```
def __len__(self):
    return len(self.labels)
```
Length Method: Returns the total number of images.
```
def __getitem__(self, idx):
    path = self.images[idx]
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (self.height, self.width))
    image = self.transform(image)
    label = self.labels[idx]
    return image, label
```
Get Item Method: Retrieves an image and its label based on the index. It reads the image, converts it to RGB, resizes it, applies transformations, and returns the processed image and label.

**3. Argument Parsing**
```
def get_args():
    parser = argparse.ArgumentParser(description="Animal classifier")
    # Argument definitions...
    args = parser.parse_args()
    return args
```
Purpose: Defines a function to parse command-line arguments for the script, allowing customization of dataset paths, training parameters, and more.

**4. Plotting Confusion Matrix**
```
def plot_confusion_matrix(writer, cm, class_names, epoch):
    # Plotting logic...
```
Purpose: This function visualizes the confusion matrix and logs it to TensorBoard. It normalizes the matrix, formats the text color based on the value, and adds labels.

**5. Training Function**

*Initial Setup*
```
def train():
    # Change all to args here...
    num_epochs = 75  # args.num_epochs
    # Other parameters...
```
Purpose: Main function to handle the training process. Initializes various parameters, including epochs, batch size, learning rate, and paths.

*Directory Creation*
```
if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
```
Purpose: Creates directories for saving TensorBoard logs and model checkpoints if they don't exist.

*Data Loading*
```
train_datasets = Datasets(dataset_path, True, height, width, size=None)
train_dataloader = DataLoader(
    train_datasets,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    drop_last=True
)
```
Purpose: Initializes the training dataset and DataLoader to manage batch loading. A similar process is followed for the validation dataset.

*Model Setup*
```
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
del model.fc
model.fc = nn.Linear(in_features=in_features, out_features=len(train_datasets.categories), bias=True)
```
Purpose: Loads a pre-trained ResNet-18 model and modifies the final layer to match the number of output classes.

*Training Loop*
```
for epoch in range(start_epoch, num_epochs):
    model.train()
    # Training logic...
    model.eval()
    # Validation logic...
```
Purpose: Main loop for training and validating the model across the specified number of epochs.

**6. Saving Checkpoints and Best Accuracy**
```
saved_data = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch + 1,
    "best_acc": best_acc,
}
checkpoint = os.path.join(checkpoint_path, "last.pt")
t.save(saved_data, checkpoint)
```
Purpose: Saves the model state, optimizer state, current epoch, and best accuracy to a checkpoint file.
```
if accuracy > best_acc:
    bestpoint = os.path.join(checkpoint_path, "best.pt")
    t.save(saved_data, bestpoint)
    best_acc = accuracy
```
Purpose: If the current accuracy exceeds the best recorded accuracy, it saves this state as the best checkpoint.

**7. Main Execution**
```
if __name__ == '__main__':
    train()
```
Purpose: Executes the train() function if the script is run as the main program.

**8. Test Function**
```
def test(image_path):
```
Purpose: Defines a function to test the classification of an image.

*Category Definitions and Device Setup*
```
  categories = ["cat", "cow", "dog", "sheep", "elephant", "butterfly", "squirrel", "horse", "chicken", "spider"]
  checkpoint_path = "/content/drive/MyDrive/checkpoint"
  device = t.device("cuda" if t.cuda.is_available() else "cpu")
  print("Device: ", device)
```
* Categories: Defines the list of possible classes for classification.
* Checkpoint Path: Specifies where the model checkpoints are saved.
* Device Setup: Checks if a GPU is available and sets the computation device accordingly.

*Model Initialization*
```
  model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
  in_features = model.fc.in_features
  del model.fc
  model.fc = nn.Linear(in_features=in_features, out_features=len(categories), bias=True)
```
* Loading the Model: Loads the pre-trained ResNet-18 model.
* Modifying the Final Layer: Removes the original fully connected layer and replaces it with a new one that outputs the number of categories defined earlier.

*Load the Checkpoint*
```
  model.to(device)
  checkpoint = os.path.join(checkpoint_path, "best.pt")
  saved_data = t.load(checkpoint, map_location=device)
  model.load_state_dict(saved_data["model"])
  model.eval()
```
* Loading Weights: Moves the model to the appropriate device and loads the best-performing model weights from the checkpoint.
* Evaluation Mode: Sets the model to evaluation mode, which disables dropout and batch normalization updates.

**9. Image Preprocessing**
```
  mean = np.array([0.485, 0.546, 0.406])
  std = np.array([0.229, 0.224, 0.225])

  ori_image = cv2.imread(image_path)
  image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (224, 224))
  transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
  image_tensor = transform(image).unsqueeze(0).to(device)
```
* Normalization Parameters: Defines the mean and standard deviation used in the normalization step.
* Image Loading and Processing:
* Reads the image from the specified path.
* Converts the image from BGR (OpenCV's format) to RGB.
* Resizes the image to 224x224 pixels (the expected input size for ResNet).
* Transformations: Applies the defined transformations (to tensor and normalization) and adds a batch dimension (using unsqueeze(0)).

**10. Model Prediction**
```
  with t.no_grad():
      output = model(image_tensor) 
      predicted_class_index = t.argmax(output, dim=1).item()
      prob = nn.Softmax(dim=1)(output)[0, predicted_class_index].item()
```
* No Gradient Calculation: Uses torch.no_grad() to disable gradient calculations, reducing memory consumption during inference.
* Model Output: Passes the preprocessed image tensor through the model to get predictions.
* Predicted Class Index: Finds the index of the class with the highest probability.
* Probability Calculation: Applies softmax to the output logits to get the probability of the predicted class.

**11. Visualization**
```
  predicted_class = categories[predicted_class_index]

  plt.imshow(image)
  plt.title(f"{predicted_class} ({prob * 100:.2f}%)")
  plt.axis('off')
  plt.show()
```
* Class Name: Retrieves the predicted class name using the predicted index.
* Plotting: Displays the original image with a title showing the predicted class and its probability. The axes are turned off for a cleaner look.

**12. Warnings Filter**
```
import warnings
warnings.filterwarnings("ignore")
Suppress Warnings: Ignores any warnings that might arise during execution, which can help keep the output clean.
```
**13. Main Execution Block**
```
if __name__ == "__main__":
  test("/images.webp")
```
Execution: Calls the test() function with a sample image path to run the classification. Make sure to replace "/images.webp" with the actual path of the image you want to test.
