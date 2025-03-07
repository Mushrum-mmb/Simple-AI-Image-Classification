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
- [License](#%EF%B8%8F-license)

### 🚀 About

This AI application performs image classification using deep learning. Trained on my private datasets with the ResNet-18 model, it accurately predicts various animal categories. Users can upload images and receive predictions along with confidence scores.

* Author: [Mushrum-mmb](https://github.com/Mushrum-mmb/)
* Model: [resnet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)

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

To run this application locally, ensure you have the following dependencies installed:
```bash
pip install torch torchvision gradio opencv-python scikit-learn matplotlib tensorboard tqdm
```
### ▶️ Usage
Clone the repository:
```bash
git clone https://github.com/Mushrum-mmb/Simple-Deep-Learning-For-Images-Classification
cd Simple-Deep-Learning-For-Images-Classification
```
Launch the application:
```bash
python run.py
```
Open the provided link in your browser to access the interface.
### 👍 How It Works

The application uses a pre-trained ResNet-18 model to analyze uploaded images. The model is fine-tuned on specific animal classes, providing accurate classification results based on the input.

### ©️ License
This project is licensed under the MIT License. See the LICENSE file for details.
