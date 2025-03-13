"""
In this script, we will load the data and start training it, as we are creating an AI for image classification, which is a basic deep learning AI.
I will use transfer learning to optimize the model and reduce training time.
As usual, we will run the script in CMD or terminal, but I am using Google Colab to train this model, which is why I haven't created a function to parse command-line arguments.
However, I think I will need it, and you might too.
Therefore, in this script, I will create a function to parse command-line arguments.
First import:

"""
import argparse


# Defines a function to parse command-line arguments for the script.
def get_args():
  parser = parser.ArgumentParser(description= "Animal Classifier")

  parser
