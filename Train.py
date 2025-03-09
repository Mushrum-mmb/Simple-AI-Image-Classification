from torch.utils.data import Dataset, DataLoader
import torch as t
import torch.nn as nn
import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
# from torchinfo import summary
from torchvision.models import resnet50, ResNet50_Weights
from Datasets import Datasets
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
import argparse

#Defines a function to parse command-line arguments for the script.
def get_args():
    parser = argparse.ArgumentParser(description="Animal classifier")
    parser.add_argument("--dataset-path", "-d", type=str, help="path to dataset")
    parser.add_argument("--tensorboard-path", "-o", type=str, help="tensorboard folder")
    parser.add_argument("--checkpoint-path", "-c", type=str, help="checkpoint folder")
    parser.add_argument("--resume-training", "-r", type=bool, default=False, help="Continue training or not")
    parser.add_argument("--height", "-h", type=int, default=224, help="height of all images")
    parser.add_argument("--width", "-w", type=int, default=224, help="width of all images")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="batch size of training procedure")
    parser.add_argument("--num-epochs","-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning-rate", "-l",type=float, default=1e-3, help="learning rate of optimizer")
    parser.add_argument("--num-train", "-t", type=int, help="number of train images")
    parser.add_argument("--num-val", "-v", type=int, help="number of test images")

    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap="OrRd")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def train():
  ####################################################################
  #chagne all to args here
  num_epochs =  75  #args.num_epochs
  batch_size = 32   #args.batch_size
  learning_rate = 0.001   #args.learning_rate
  num_train = None    #args.num_train
  num_val = None     #args.num_val
  resume_training = False #args.resume_training
  height = 224 #args.height
  width = 224 #args.width
  dataset_path = "/content/animals_v2" #args.dataset_path
  tensorboard_path = "/content/drive/MyDrive/tensorboard" #args.tensorboard_path
  checkpoint_path = "/content/drive/MyDrive/checkpoint" #args.checkpoint_path
  #####################################################################
  if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
  if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
  writer = SummaryWriter(tensorboard_path)
  #####################################################################
  train_datasets = Datasets(dataset_path, True, height, width, num_train)
  train_dataloader = DataLoader(
      train_datasets,
      batch_size=batch_size,
      shuffle=True,
      num_workers=2,
      drop_last=True)
  val_datasets = Datasets(dataset_path, False, height, width, num_val)
  val_dataloader = DataLoader(
      val_datasets,
      batch_size=batch_size,
      shuffle=False,
      num_workers=2,
      drop_last=False)
  ###########################################################################
  num_iters = len(train_dataloader)
  device = t.device("cuda" if t.cuda.is_available() else "cpu")
  print("Device: ",device)
  print("len_train_data: ", len(train_datasets))
  print("len_val_data: ", len(val_datasets))

  model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
  in_features = model.fc.in_features
  del model.fc
  model.fc = nn.Linear(in_features=in_features, out_features=len(train_datasets.categories), bias=True)

  model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

  if resume_training:
    checkpoint = os.path.join(checkpoint_path, "last.pt")
    saved_data = t.load(checkpoint, map_location=device)
    model.load_state_dict(saved_data["model"])
    optimizer.load_state_dict(saved_data["optimizer"])
    start_epoch = saved_data["epoch"]
    accuracy = saved_data["accuracy"]
    #load and print best acc
    bestpoint = os.path.join(checkpoint_path, "best.pt")
    saved_best_point = t.load(bestpoint,  map_location=device)
    goat = saved_best_point["accuracy"]
    print(f"Best Accuracy Now: {goat:.4f}")
  else:
        start_epoch = 0
        accuracy = -1
        goat= -1

  print("Start training....")
  print("Start at epoch: ", start_epoch)
  for epoch in range(start_epoch, num_epochs):
    #training
    model.train()
    total_losses = []
    progress_bar = tqdm.tqdm(train_dataloader, colour="yellow")
    for iter , (images, labels) in enumerate(progress_bar):
      #forward pass
      images = images.to(device)
      labels = labels.to(device)
      output = model(images)
      loss = criterion(output, labels)

      total_losses.append(loss.item())
      avg_loss = np.mean(total_losses)
      progress_bar.set_description(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss:0.4f}")
      writer.add_scalar("Train/Loss", loss, global_step=epoch*num_iters+iter)
      writer.add_scalar("Avg_Train/Avg_Loss", avg_loss, global_step=epoch*num_iters+iter)

      #backward pass
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


    #validation
    model.eval()
    total_losses = []
    all_labels = []
    all_predictions = []
    progress_bar = tqdm.tqdm(val_dataloader, colour="green", desc="Validation")
    # with torch.inference_mode():    # From pytorch 1.9
    with t.no_grad():
      for iter, (images, labels) in enumerate(progress_bar):
        # forward pass
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)   # shape [batch_size, num_classes]
        loss = criterion(output, labels)
        prediction = t.argmax(output, dim=1)

        total_losses.append(loss.item())
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(prediction.cpu().tolist())



    avg_loss = np.mean(total_losses)
    writer.add_scalar("Val/Loss", avg_loss, global_step=epoch)
    accuracy = accuracy_score(all_labels, all_predictions)
    writer.add_scalar("Val/Accuracy", accuracy, global_step=epoch)
    print(f"Epoch: {epoch+1}, Validation Loss: {avg_loss:.4f}, Validation Accuracy Now: {accuracy:.4f}")
    plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), train_datasets.categories, epoch)
    #save architecture
    saved_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch+1,
            "accuracy": accuracy,
        }
    checkpoint = os.path.join(checkpoint_path, "last.pt")
    t.save(saved_data, checkpoint)
    if epoch > 0:
      #load and print best acc
      bestpoint = os.path.join(checkpoint_path, "best.pt")
      saved_best_point = t.load(bestpoint,  map_location=device)
      goat = saved_best_point["accuracy"]
      print(f"Best Accuracy Before: {goat:.4f}")
    if accuracy > goat:
      bestpoint = os.path.join(checkpoint_path, "best.pt")
      t.save(saved_data, bestpoint)
  writer.close()
if __name__ == '__main__':
    train()
