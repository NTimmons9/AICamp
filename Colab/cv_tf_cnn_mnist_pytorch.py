# -*- coding: utf-8 -*-
"""Copy of CV_TF_CNN_MNIST_PyTorch.ipynb

Provided by AI Camp

<h1>Getting the Environment Set Up</h1>

We will be using PyTorch to create and use our neural networks. Throughout this notebook, we will see just how easy it is to make a working neural network that has a surprisingly high accuary! This is a great post to refer https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118

Before we get started, remember importing the libraries and packages we need. These tools will make our job a lot easier.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets


import matplotlib.pyplot as plt
import numpy as np

"""Make sure you are using GPU, which will accelerate the training process"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

"""<h1>Managing our Images</h1>

### Download our images for training and testing
"""

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

"""It shows that we have 60,000 images for training. Each image is 28 pixels by 28 pixels"""

train_images = mnist_trainset.data
train_images.shape

"""Correspondingly, we have 60,000 label. One for each image."""

train_labels = mnist_trainset.targets
train_labels.shape

"""Similar, we have 10,000 images to test our model after training."""

test_images = mnist_testset.data
test_images.shape

test_labels = mnist_testset.targets
test_labels.shape

"""Let's check how one image is stored and recognized by computer."""

train_images[0]

"""That wall of numbers isn't very easy to understand to our brains however. Lets go ahead and display a few of the images from the dataset with matplotlib. If you want to read more about matplot lib, you can go here: https://matplotlib.org/stable/users/index"""

figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(60000, size=(1,)).item()
    img, label = mnist_trainset.data[sample_idx], mnist_trainset.targets[sample_idx].item()
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

"""## Create Dataset

After having a general understanding of our data, we can ask our model to learn the data and make predictions! To get start, we first need to convert our data into a format that can be easily accessible by our model. **Dataloader** is our friend here.

Essentially, there is nothing special here. We just store our data in a convenient way for model training
"""

train_images = train_images.reshape(-1, 1, 28, 28)
test_images = test_images.reshape(-1, 1, 28, 28)

from torch.utils.data import Dataset
class ImageDataset(Dataset):
  def __init__(self,img, label):
    self.img = img
    self.label = label

  def __len__(self):
    return len(self.label)
  def __getitem__(self, idx):
    return self.img[idx], self.label[idx]

from torch.utils.data import DataLoader
train_dataset = ImageDataset(train_images, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=1)

"""<h1>Creating Our Model</h1>

PyTorch lets us create a model super easily. You can treat the model-building process as building LEGO. Different module may have different functions. We need to find what we need and compile them together. Here is what we used for the CNN.



*   Conv2d: Add a Convolutional layer
*   ReLU: Activation Function

*   MaxPool2d: Pooling Layer

*   Linear: Fully Connected Layer
"""

# The parameters in each of these Conv2D layers are as follows:
# nn.Conv2D(In_channel ,Output_channl, Kernel Size)

def create_model():
  model = nn.Sequential(
      nn.Conv2d(1, 28, (3, 3), padding=1),
      nn.ReLU(),
      nn.MaxPool2d((2, 2)),

      nn.Conv2d(28, 56, (3, 3), padding=1),
      nn.ReLU(),
      nn.MaxPool2d((2, 2)),

      nn.Conv2d(56, 56, (3, 3), padding=1),
      nn.ReLU(),
      nn.MaxPool2d((2, 2)),

      nn.Flatten(),
      nn.Linear(504, 64),
      nn.ReLU(),
      nn.Linear(64, 10)
  )
  return model

"""After defining the model architecture, we define 2 other things for model training:


*   Optimizer: How we want our model to update itself after each step
*   Creiteron (Loss function): How wrong our model perform


"""

model = create_model()
# The optimizer is just an algorithm that helps the AI learn faster
# The loss is Sparse Categorical Crossentropy, the name is scary but it just means that
# the outputs are converted to percentages and then compared to the expected output
# Metrics = accuracy means that we only care about how close the answer was to the actual answer

optimizer = optim.Adam(model.parameters(), lr=1e-5)
criteron = nn.CrossEntropyLoss()

model.to(device)

"""<h1>Training our Model</h1>

Now, we are prepared to train our model. PyTorch makes training easy. All we need to do is give it the input data and the expected outputs with the number of epochs to train for and it will handle the rest!
"""

def train_model(model, train_loader, loss_fn, optimizer, epochs, test_images, test_labels):
  for i in range(epochs):
    for img, label in train_loader:
      img = img.to(device)
      img = img.to(torch.float)
      label = label.to(device)

      pred = model(img)

      loss = loss_fn(pred, label)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    test_images = test_images.to(torch.float).to(device)
    pred = model(test_images)
    digit = torch.argmax(pred, dim=1)
    test_labels = test_labels.to(device)
    acc = torch.sum(digit == test_labels)/len(test_labels)
    print(f"Epoch {i+1}: loss: {loss}, test accuracy: {acc}")

train_model(model, train_loader, criteron, optimizer, 10, test_images, test_labels)

"""<h1>Evaluating our Model</h1>

Evaluating out model with PyTorch is super easy. We can see blow that we were able to get about a 90% accuracy on images that our model has never seen before!
"""

test_images = test_images.to(torch.float).to(device)
pred = model(test_images)
digit = torch.argmax(pred, dim=1)
test_labels = test_labels.to(device)
acc = torch.sum(digit == test_labels)/len(test_labels)
print(f"Test accuracy: {acc}")

"""Now, let's plot the confusion matrix to see where our model does well and where it makes mistakes."""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(test_labels.cpu().detach().numpy(), digit.cpu().detach().numpy())
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

"""## Further Exploration

Now you can try to give an image to the model and see whether it can recoginize it correctly.
"""

sample_idx = torch.randint(10000, size=(1,)).item()

image = test_images[sample_idx].reshape(1, 1, 28, 28)
label = test_labels[sample_idx]

pred = torch.argmax(model(image), dim=1).item()

plt.title(f"index: {sample_idx}, true:{label}, Predict:{pred}")
plt.axis("off")
plt.imshow(mnist_testset.data[sample_idx].squeeze(), cmap="gray")

