import gradio as gr
import torch as t
import torch.nn as nn
import numpy as np
import cv2
import os
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.models import resnet18, ResNet18_Weights
import argparse 

# Set up argument parser
parser = argparse.ArgumentParser(description="Animal Classifier using ResNet-18")
parser.add_argument('--checkpoint', type=str, required=True, 
                    help='Path to the model checkpoint (e.g., chechpoint/)')

args = parser.parse_args()


# Load the model once
categories = ["cat", "cow", "dog", "sheep", "elephant", "butterfly", "squirrel", "horse", "chicken", "spider"]

#change the checkpoint path here
######################################################################################
checkpoint_path = args.checkpoint
######################################################################################
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Initialize model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
del model.fc
model.fc = nn.Linear(in_features=in_features, out_features=len(categories), bias=True)
model.to(device)

# Load model weights
checkpoint = os.path.join(checkpoint_path, "best.pt")
saved_data = t.load(checkpoint, map_location=device)
model.load_state_dict(saved_data["model"])
model.eval()

# Define preprocessing
mean = np.array([0.485, 0.546, 0.406])
std = np.array([0.229, 0.224, 0.225])
transform = Compose([ToTensor(), Normalize(mean=mean, std=std)])

def classify_image(image):
    # Convert image from Gradio format (numpy array) to the format needed for the model
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (224, 224))
    image_tensor = transform(image).unsqueeze(0).to(device)

    with t.no_grad():
        output = model(image_tensor)
        predicted_class_index = t.argmax(output, dim=1).item()
        prob = nn.Softmax(dim=1)(output)[0, predicted_class_index].item()

    predicted_class = categories[predicted_class_index]
    return f"{predicted_class} ({prob * 100:.2f}%)"

# Create Gradio Interface
iface = gr.Interface(fn=classify_image, inputs="image", outputs="text", title="Animal Classifier",
                     description="Upload an image of an animal to get its classification (cat, cow, dog, sheep, elephant, butterfly, squirrel, horse, chicken, spider) <br> This project made by Phuong Nam aka Namush =))")

# Launch the interface
iface.launch(share=True)
