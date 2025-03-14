"""
In this script, we will load the data and start training it, as we are creating an AI for image classification, which is a basic deep learning AI.
I will use transfer learning to optimize the model and reduce training time.
As usual, we will run the script in CMD or terminal, but I am using Google Colab to train this model, which is why I haven't created a function to parse command-line arguments.
However, I think I will need it, and you might too.
Therefore, in this script, I will create a function to parse command-line arguments.
First import:

"""
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from Datasets import AnimalDatasets
from torch.utils.data import DataLoader
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Defines a function to parse command-line arguments for the script.
def get_args():
  parser = parser.ArgumentParser(description= "Animal Classifier")

  parser.add_argument("--num-epochs","-e", type=int, default=75, help="Number of epochs")
  parser.add_argument("--batch-size", "-b", type=int, default=64, help="batch size of training procedure")
  parser.add_argument("--height", "-h", type=int, default=224, help="height of all images")
  parser.add_argument("--width", "-w", type=int, default=224, help="width of all images")
  parser.add_argument("--num-train", "-t", type=int, default=None, help="number of train images")
  parser.add_argument("--num-val", "-v", type=int, default=None, help="number of test images")
  parser.add_argument("--learning-rate", "-l",type=float, default=1e-3, help="learning rate of optimizer")

  parser.add_argument("--resume-training", "-r", type=bool, default=False, help="Continue training or not")
  
  parser.add_argument("--dataset-path", "-d", type=str, help="path to dataset")
  parser.add_argument("--tensorboard-path", "-o", type=str, help="tensorboard folder")
  parser.add_argument("--checkpoint-path", "-c", type=str, help="checkpoint folder")
    
  args = parser.parse_args()
  return args

# Defines a train function.
def train():  #def train(args):
  # Initialize num_epochs, batch_size, height, width, num_train, num_val, learning_rate
  num_epochs = 75 #args.num_epochs
  batch_size = 64 #args.batch_size
  height = 224 #args.height
  width = 224 #args.width
  num_train = None #args.num_train
  num_val = None #args.num_val
  learning_rate = 1e-3 #args.learning_rate
  # Boolean resume_training
  resume_training = False
  # Add path of dataset, tensorboard and checkpoint
  dataset_path = "/content/drive/MyDrive/AnimalDataset" #args.dataset_path
  tensorboard_path = "/content/drive/MyDrive/tensorboard" #args.tensorboard_path
  checkpoint_path = "/content/drive/MyDrive/checkpoint" #args.checkpoint_path
  
  # Create the TensorBoard and checkpoint paths if they do not exist.
  if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
  if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

  # Visualization within the TensorBoard UI.
  writer = SummaryWriter(tensorboard_path)
  
  # Retrieve the training and validation datasets. Use a DataLoader to retrieve images in batch size, shuffle them, and drop the last batch if needed.
  train_datasets = AnimalDatasets(dataset_path, True, height, width, num_train)
  val_datasets = AnimalDatasets(dataset_path, False, height, width, num_val)
  print(f"len_train_data: {len(train_datasets)}")
  print(f"len_val_data: {len(val_datasets)}")
  train_dataloader = DataLoader(
    train_datasets,
    batch_size,
    shuffle = True,
    num_workers = 2,
    drop_last = True
  )
  val_dataloader = DataLoader(
    val_datasets,
    batch_size,
    shuffle = False,
    num_workers = 2,
    drop_last = False
  )
  
  # Use PyTorch to set the device to CUDA if available; otherwise, set it to CPU.
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")
  
  # In this section, we will choose the model. Since I am using transfer learning, I will select ResNet50 and its weights.
  model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
  in_features = model.fc.in_features # Remain the in_features of Dense Layer
  del model.fc # Delete it
  model.fc = nn.Linear(in_features=in_features, out_features=len(train_datasets.categories), bias=True) # Modify it to remove the fully connected layers, ensuring the output length matches the number of categories.
  # Bring model into device
  model.to(device)
  
  # Loss and Optimizer Setup
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
  # Create Resume Training Logic
  if resume_training:
    # Load last point
    lastpoint = os.path.join(checkpoint_path, 'last.pt')
    saved_last_data = torch.load(lastpoint, map_location=device)
    model.load_state_dict(saved_last_data['model'])
    optimizer.load_state_dict(saved_last_data['optimizer'])
    start_epoch = saved_last_data['epoch']
    accuracy = saved_last_data['accuracy']
    # Load best point
    bestpoint = os.path.join(checkpoint_path, 'best.pt')
    saved_best_data = torch.load(bestpoint, map_location=device)
    goat = saved_best_data['accuracy']
    print(f"Best Accuracy now: {goat:.4f}") # Print the best accuracy now
  else:
    start_epoch = 0
    accuracy = -1
    goat = -1

  # Create training loop
  print(f"Start training at epoch {start_epoch}")
  for epoch in range(start_epoch, num_epochs):
    # Training phase
    model.train()
    # Initalize counter total losses and progress bar for per training phase
    total_losses = []
    progress_bar = tqdm(train_dataloader, colour="yellow")
    
    for iter, (images, labels) in enumerate(progress_bar):
      # Forward pass
      images = images.to(device)
      output = model(images)
      # Calculate loss, total_losses and avg_loss
      labels = labels.to(device)
      loss = criterion(output, labels)
      total_losses.append(loss.item())
      avg_loss = np.mean(total_losses) # We will observe the average loss for per batch size
      
      # Visualization
      progress_bar.set_description(f"Training phase at epoch: {epoch+1}/{num_epochs}, Avg_Loss: {avg_loss:.4f}")
      writer.add_scalar("Train/Loss", loss, global_step= epoch*len(train_dataloader)+iter)
      writer.add_scalar("Avg_Train/Avg_Loss", avg_loss, global_step= epoch*len(train_dataloader)+iter)

      # Backward pass
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # Validation phase
    model.eval()
    # Initalize counter total losses, all labels, all predictions and progress bar for validation phase
    total_losses = []
    all_labels = []
    all_predictions = []
    progress_bar = tqdm(val_dataloader, colour="green")

    with torch.no_grad():
      for iter, (images, labels) in enumerate(progress_bar):
        # Forward pass
        images = images.to(device)
        output = model(images)
        # Calculate loss, total_losses, avg_loss and prediction
        labels = labels.to(device)
        loss = criterion(output, labels)
        total_losses.append(loss.item())
        avg_loss = np.mean(total_losses) # We will observe the average loss for per batch size
        prediction = torch.argmax(output, dim=1)
        '''
        To collecting all labels and all predictions
        We will move the labels tensor and predictions tensor from the GPU to the CPU and converts it into a standard Python list.
        This is necessary because Python lists cannot hold GPU tensors.
        '''
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(prediction.cpu().tolist())

        # Visualization
        progress_bar.set_description(f"Validation phase at epoch: {epoch+1}/{num_epochs}, Avg_Loss: {avg_loss:.4f}")

    avg_loss = np.mean(total_losses) # We will observe the average loss after evaluating all batches
    # Visualization
    avg_loss = np.mean(total_losses)
    writer.add_scalar("Val/Loss", avg_loss, global_step=epoch)
    accuracy = accuracy_score(all_labels, all_predictions)
    writer.add_scalar("Val/Accuracy", accuracy, global_step=epoch)
    print(f"Epoch: {epoch+1}, Validation Loss: {avg_loss:.4f}, Validation Accuracy now: {accuracy:.4f}")
    # Visualize the confusion matrix of a classification model, which helps you understand how well the model is performing across different classes
    plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), train_datasets.categories, epoch)

    # Checkpoint saving 
    saved_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch+1,
            "accuracy": accuracy,
        }
    checkpoint = os.path.join(checkpoint_path, "last.pt")
    torch.save(saved_data, checkpoint)

    # Load and print the best accuracy after training the first epoch.
    if epoch > 0:
      bestpoint = os.path.join(checkpoint_path, 'best.pt')
      saved_best_data = torch.load(bestpoint, map_location=device)
      goat = saved_best_data['accuracy']
      print(f"Best Accuracy before: {goat:.4f}")
    # Compare the accuracy with the best accuracy; if it is better, we will save it as best.pt.
    if accuracy > goat:
      bestpoint = os.path.join(checkpoint_path, "best.pt")
      torch.save(saved_data, bestpoint)

  # Close the writer
  writer.close()

# Defines a function named plot_confusion_matrix, which is designed to visualize a confusion matrix using Matplotlib.
def plot_confusion_matrix(writer, cm, class_names, epoch):
  # Create a figure
  figure = plt.figure(figsize=(20, 20))
  # Plotting the Confusion Matrix
  plt.imshow(cm, interpolation='nearest', cmap="OrRd")
  plt.title("Confusion matrix")
  plt.colorbar()
  # Setting tick marks
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Normalize the confusion matrix
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Adding text annotation
  threshold = cm.max() / 2.
  for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
          color = "white" if cm[i, j] > threshold else "black"
          plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

  # Finalizing the plot
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  # Logging the figure
  writer.add_figure('confusion_matrix', figure, epoch)

if __name__ == '__main__':
  train()
