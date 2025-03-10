import gradio as gr
import torch as t
import torch.nn as nn
import numpy as np
import cv2
import os
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.models import resnet50, ResNet50_Weights
import argparse 

# Set up argument parser
parser = argparse.ArgumentParser(description="Animal Classifier using ResNet-18")
parser.add_argument('--checkpoint', type=str, required=True, 
                    help='Path to the model checkpoint (e.g., chechpoint/)')

args = parser.parse_args()


# Load the model once
categories = ['elephant', 'cat', 'horse', 'spider', 'dog', 'chicken', 'butterfly', 'cow', 'sheep', 'squirrel']

#change the checkpoint path here
######################################################################################
checkpoint_path = args.checkpoint
######################################################################################
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Initialize model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
del model.fc
model.fc = nn.Linear(in_features=in_features, out_features=len(categories), bias=True)
model.to(device)

# Load model weights
bestpoint = os.path.join(checkpoint_path, "best.pt")
saved_data = t.load(bestpoint, map_location=device)
best = saved_data["accuracy"]
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
                     description=f"Upload an image of an animal to get its classification (cat, cow, dog, sheep, elephant, butterfly, squirrel, horse, chicken, spider) <br> This project made by Phuong Nam aka Namush =)) <br> The accuracy of the model is {best*100:.1f}% ")

# Launch the interface
iface.launch(share=True)
