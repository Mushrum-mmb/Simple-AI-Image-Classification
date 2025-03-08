import torch as t
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights
import os
import cv2
import numpy as np
from torchvision.transforms import ToTensor, Normalize, Compose
import matplotlib.pyplot as plt

def test(image_path):
  categories = ["cat", "cow", "dog", "sheep", "elephant", "butterfly", "squirrel", "horse", "chicken","spider"]
  checkpoint_path = "/content/drive/MyDrive/checkpoint"
  device = t.device("cuda" if t.cuda.is_available() else "cpu")
  print("Device: ",device)
  # model = EfficientNet_B7(len(categories))
  model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
  in_features = model.fc.in_features
  del model.fc
  model.fc = nn.Linear(in_features=in_features, out_features=len(categories), bias=True)

  model.to(device)
  checkpoint = os.path.join(checkpoint_path, "best.pt")
  saved_data = t.load(checkpoint, map_location=device)
  model.load_state_dict(saved_data["model"])
  model.eval()

  mean = np.array([0.485, 0.546, 0.406])
  std = np.array([0.229, 0.224, 0.225])

  ori_image = cv2.imread(image_path)
  image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (224,224))
  transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
  image_tensor = transform(image).unsqueeze(0).to(device)

  with t.no_grad():
      output = model(image_tensor) 
      predicted_class_index = t.argmax(output, dim=1).item()
      prob = nn.Softmax(dim=1)(output)[0, predicted_class_index].item()

  predicted_class = categories[predicted_class_index]

  plt.imshow(image)
  plt.title(f"{predicted_class} ({prob * 100:.2f}%)")
  plt.axis('off')
  plt.show()

import warnings
warnings.filterwarnings("ignore")
if __name__ == "__main__":
  test("/images.webp")
