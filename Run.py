"""
In this script, we will create an interface that can test our model and share the link so others can use it too.
First import:
- import torch: Main library for building and training neural networks in PyTorch.
- import torch.nn as nn: Contains classes for building neural network layers in PyTorch.
- import os: Interacts with the operating system for file and directory operations.
- import numpy as np: Supports numerical computations and array operations.
- from torchvision.transforms import ToTensor, Normalize, Compose: For image preprocessing (converting to tensors, normalization, and chaining transformations).
- import cv2: For computer vision and image processing tasks.
- import matplotlib.pyplot as plt: Creates visualizations for data analysis.
- import gradio as gr: Creates user-friendly web interfaces for machine learning models.
- import warnings: is used to issue and manage warning messages in Python.
"""
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import os
import numpy as np
from torchvision.transforms import ToTensor, Normalize, Compose
import cv2
import matplotlib.pyplot as plt
import gradio as gr
import warnings
"""
The warning we are encountering indicates that we are using torch.load with the default parameter weights_only=False.
This setting can be risky because it allows the loading of arbitrary Python objects, which could potentially execute malicious code during the unpickling process.
However, we are not only loading the model weights but also other components like the optimizer state, accuracy, and additional metadata.
Therefore, setting weights_only=True may not be suitable for our use case.
We will use the warnings module to ignore warnings if they become too messy.
In a future release of PyTorch, the default value of weights_only will change to True, so the code in the future may include a parameter like weights_only=False.
"""
warnings.filterwarnings("ignore")

# Initialize the constantly categories.
categories = ['dog','cat','cabypara','hamster','parrot','pufferfish']
# Set the checkpoint_path
checkpoint_path = "/content/drive/MyDrive/checkpoint"
# Use PyTorch to set the device to CUDA if available; otherwise, set it to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
  
# In this section, we will choose the model. Since I am using transfer learning, I will select ResNet50 and its weights.
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features # Remain the in_features of Dense Layer
del model.fc # Delete it
model.fc = nn.Linear(in_features=in_features, out_features=6, bias=True) # Modify it to remove the fully connected layers, ensuring the output length matches the number of categories.
# Bring model into device
model.to(device)

# Load the weights and accuracy in the best.pt
bestpoint = os.path.join(checkpoint_path, "best.pt")
saved_data = torch.load(bestpoint, map_location=device)
model.load_state_dict(saved_data["model"])
# Load and print the accuracy
best = saved_data["accuracy"]
print(f"Accuracy: {best:.4f}")
# Switch the model to evaluate mode
model.eval()
# Initialize the mean and standard deviation for normalization. Compose the ToTensor() and Normalize() functions to initialize the transforms.
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
transform = Compose([
  ToTensor(),
  Normalize(mean, std)
])

# Define a classify image function
def classify_image(image):
  # Retrieve an image, resize and transform it to let the model can get it.
  image = cv2.resize(image, (224,224))
  # ToTensor() will be (3, 224, 224) and after unsqueeze(0), the shape will be (1, 3, 224, 224), which is suitable for model input.
  image_tensor = transform(image).unsqueeze(0).to(device) 

  # Start predicting and calculate the scores
  with torch.no_grad():
    output = model(image_tensor)
    pred_index = torch.argmax(output, dim=1).item()
    scores = nn.Softmax(dim=1)(output)
    score = scores[0, pred_index].item()

  pred_cate = categories[pred_index]
  return f"{pred_cate} ({score * 100:.2f}%)"

# Create Gradio Interface
iface = gr.Interface(fn=classify_image,
                      inputs="image",
                      outputs="text",
                      title="Animal Classifier",
                      description=f"Upload an image of an animal to get its classification (cat, dog, cabypara, parrot, hamster, pufferfish) <br> This project made by Phuong Nam aka Namush =)) <br> The accuracy of the model is {best*100:.1f}% <br> Note: Đã tối ưu mô hình và thu hẹp số con vật, cơ mà accuracy trên là test với bộ validation mà nó cao vãi mèo nên sú <br> Cần lắm ae test nghiêm túc =))")
# Launch the interface
iface.launch(share=True)
