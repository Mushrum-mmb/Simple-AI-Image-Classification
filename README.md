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

This AI application performs image classification using deep learning. Trained on my private datasets with the ResNet-50 model, it accurately predicts various animal categories. Users can upload images and receive predictions along with confidence scores.

* Author: [Mushrum-mmb](https://github.com/Mushrum-mmb/)
* Model: [resnet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50)
* Framework: [gradio](https://www.gradio.app/)

![image](https://github.com/user-attachments/assets/0ca4a168-0f6c-4644-8068-ec4efb402332)
![image](https://github.com/user-attachments/assets/c8fb6d3e-8c98-4b5d-9d13-e81f5908a1e2)


### 🎓 Features
* Image Classification:
The application allows users to upload images for classification into specific animal categories.
* Pre-trained Model:
Utilizes a pre-trained ResNet-50 model fine-tuned for classifying various animals.
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

**1. Datasets.py:**
* Defines a custom dataset class, Datasets, that inherits from torch.utils.data.Dataset.
* It initializes with paths to image files and their corresponding labels, normalizes images, and prepares transformations for training and testing.
* Supports loading training and testing data from specified directories.
  
**2. Train.py:**
* Contains the main training loop for an animal classification model using a ResNet architecture.
* Parses command-line arguments for dataset paths, hyperparameters, and logging paths.
* Initializes the dataset, data loaders, model, loss function, and optimizer.
* Trains the model for a specified number of epochs, logging training and validation losses and accuracies to TensorBoard.
* Saves model checkpoints during training and tracks the best model based on validation accuracy.

**3. Test.py:**
* Loads a trained model and performs inference on a single image.
* Preprocesses the image, runs it through the model, and predicts the class label with the associated probability.
* Displays the original image along with the predicted class and confidence.
  
**4. Run.py:**
* Sets up a Gradio web interface for the model, enabling users to upload images for classification.
* Loads the trained model from a specified checkpoint, processes uploaded images, and returns predictions.
* Provides a user-friendly interface for real-time image classification.
