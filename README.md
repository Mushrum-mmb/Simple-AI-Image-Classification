# Simple-AI-Image-Classification

# Work in progress ⏳ The training is still not complete -_-

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
- [Google Colab Usage](#-google-colab-usage)
- [How It Works](#-how-it-works)


### 🚀 About

This AI application performs image classification using deep learning. Trained on my private datasets with the ResNet-50 model, it accurately predicts various animal categories. Users can upload images and receive predictions along with confidence scores.

* Author: [Mushrum-mmb](https://github.com/Mushrum-mmb/)
* Model: [resnet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50)
* Framework: [gradio](https://www.gradio.app/)

![image](https://github.com/user-attachments/assets/0ca4a168-0f6c-4644-8068-ec4efb402332)
![image](https://github.com/user-attachments/assets/c8fb6d3e-8c98-4b5d-9d13-e81f5908a1e2)
## The accuracy now is 99.91% (kinda sus, as this accuracy is calculated from the validation datasets).
![individualImage](https://github.com/user-attachments/assets/15509cc8-ad6e-4a65-aa44-0297fbffdb9f)


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
* Easy Usage:
I am training this AI in Google Colab, so you can use and test it there. Google Colab is suitable for those who have a low-spec device like mine.

### ⬇️ Installation
***Ensure that you have already installed Git and set up your Python environment.***

Note: For Google Colab users, skip the installation below and download only the 'Training artifacts'.

To run this application locally, ensure you have opened the CMD and have the following dependencies installed:
```bash
pip install torch torchvision gradio opencv-python scikit-learn matplotlib tensorboard tqdm requests beautifulsoup4
```

To download the zip file named 'Training artifacts', click on the release tab or [click here.](https://github.com/Mushrum-mmb/Simple-AI-Image-Classification/releases/tag/Training_artifacts) 

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
Copy the path of the checkpoint folder in Training artifacts that contains best.pt; it will work like this: 

![image](https://github.com/user-attachments/assets/e7706a92-eceb-4808-b7b0-08f2f5f7fede)

Then launch the application by running run.py and paste the path of the folder containing best.pt.

#For ex: python run.py --checkpoint "C:\Users\DELL\Downloads\Training artifacts\checkpoint"
```bash
python run.py --checkpoint "your-checkpoint-path"

```
Open the provided link in your browser to access the interface. Your work will look like this after everything:

![image](https://github.com/user-attachments/assets/07360da9-aae1-4797-bfef-9f2ea7aba9a4)

### 💻 Google Colab Usage

If you can't train and run the script on your potato computer like I do, you can also run the script in Google Colab, which I am using to train this AI. :')

Just read and run the cell in Google Colab for investment purposes. If you only want to use the AI, I will provide direct tutoring.

[Click here for access my notebook](https://colab.research.google.com/drive/13yuj3zqh8ed1wi9KkUfnDeBKN0ZYgel1?usp=sharing)

First, scroll to this section:

Then run the first cell to install `Gradio`
![image](https://github.com/user-attachments/assets/85778e45-9bdf-4b05-a9d8-48efedd338f6)

Then upload the 'Training artifacts' to the /content path. This path is the default when you connect to the runtime.

![image](https://github.com/user-attachments/assets/5dde14d7-eac2-462b-bd45-a672e5d02815)
Done

![image](https://github.com/user-attachments/assets/e7ec753c-9607-42bd-80f2-4a3f944d946c)

Run the next cell.

![image](https://github.com/user-attachments/assets/29b378f6-6041-4205-a649-b2bcf07b083c)


### 👍 How It Works

**1. Collect.py:**
* Automates downloading images from Google Images based on a search query.
* Defines a function, collect_images, that takes a search term, number of images, and a directory path as input.
* Constructs a Google Images search URL with pagination, makes an HTTP request, and parses the HTML to find image URLs.
* Validates and downloads the images, saving them to the specified directory with structured filenames.

**2. Datasets.py:**
* Defines a custom dataset class, Datasets, that inherits from torch.utils.data.Dataset.
* It initializes with paths to image files and their corresponding labels, normalizes images, and prepares transformations for training and testing.
* Supports loading training and testing data from specified directories.
  
**3. Train.py:**
* Contains the main training loop for an animal classification model using a ResNet architecture.
* Parses command-line arguments for dataset paths, hyperparameters, and logging paths.
* Initializes the dataset, data loaders, model, loss function, and optimizer.
* Trains the model for a specified number of epochs, logging training and validation losses and accuracies to TensorBoard.
* Saves model checkpoints during training and tracks the best model based on validation accuracy.

**4. Test.py:**
* Loads a trained model and performs inference on a single image.
* Preprocesses the image, runs it through the model, and predicts the class label with the associated probability.
* Displays the original image along with the predicted class and confidence.
  
**5. Run.py:**
* Sets up a Gradio web interface for the model, enabling users to upload images for classification.
* Loads the trained model from a specified checkpoint, processes uploaded images, and returns predictions.
* Provides a user-friendly interface for real-time image classification.
