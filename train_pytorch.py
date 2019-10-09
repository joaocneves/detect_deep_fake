import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import argparse
import torchvision
from torchvision import transforms, datasets
import copy

freeze_layers = False
n_class = 2

# Load the model
model_conv = torchvision.models.resnet50(pretrained='imagenet')
print(model_conv)

# Lets freeze the first few layers. This is done in two stages
# Stage-1 Freezing all the layers
if freeze_layers:
  for i, param in model_conv.named_parameters():
    param.requires_grad = False

# Since imagenet as 1000 classes , We need to change our last layer according to the number of classes we have,

## If VGG16:
#print(model_conv)
#num_ftrs = model_conv.classifier[6].in_features
#model_conv.classifier[6] = nn.Linear(num_ftrs, n_class)

# If squeezenet
#model_conv.classifier[1] = nn.Conv2d(512, n_class, kernel_size=(1,1), stride=(1,1))

# If resnet50
model_conv.fc = nn.Linear(2048, n_class)


#If dataset not in format:
#import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to ratio, i.e, (.8, .2).
#split_folders.ratio(data_dir, output="dataset", seed=1337, ratio=(.7, .15, .15)) # default values

# Loading the dataloaders -- Make sure that the data is saved in following way
"""
data/
  - train/
      - class_1 folder/
          - img1.png
          - img2.png
      - class_2 folder/
      .....
      - class_n folder/
  - val/
      - class_1 folder/
      - class_2 folder/
      ......
      - class_n folder/
"""

data_dir = "/media/jcneves/DATASETS/real2fake_mixed/"
input_shape = 200
batch_size = 32
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
scale = 224
use_parallel = False
use_gpu = False
epochs = 2

data_transforms = {
        'train': transforms.Compose([
        transforms.Resize(scale),
        # transforms.RandomResizedCrop(input_shape),
        #transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
        transforms.Resize(scale),
        # transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
        ]), }

test_transforms = {
        'test': transforms.Compose([
        transforms.Resize(scale),
        transforms.ToTensor(),
        ]),}

# Train and val set
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                      data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                         shuffle=True, num_workers=4) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# test set
test_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                      test_transforms[x]) for x in ['test']}
test_dataloader = {x: torch.utils.data.DataLoader(test_datasets[x], batch_size=1, num_workers=1) for x in ['test']}

if use_parallel:
    print("[Using all the available GPUs]")
    model_conv = nn.DataParallel(model_conv, device_ids=[0])

print("[Using CrossEntropyLoss...]")
criterion = nn.CrossEntropyLoss()

print("[Using small learning rate with momentum...]")
optimizer_conv = optim.SGD(list(filter(
    lambda p: p.requires_grad, model_conv.parameters())), lr=0.001, momentum=0.9)

print("[Creating Learning rate scheduler...]")
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            running_loss_b = 0.0
            i = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                #inputs = inputs.cuda()
                #labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # print statistics
                    running_loss_b += loss.item()
                    if i % 10 == 9:  # print every 100 mini-batches
                        print('[%d, %5d] loss: %.3f acc: %.3f' %
                              (epoch + 1, i + 1, running_loss / (i*batch_size), float(running_corrects) / (i*batch_size) ))
                        running_loss_b = 0.0

                    i = i + 1

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def test_model(model, dataloaders):

    running_corrects = 0
    for inputs, labels in dataloaders['test']:
        #inputs = inputs.cuda()
        #labels = labels.cuda()

        with torch.no_grad():
            # Get model outputs and calculate correct label
            outputs = model(inputs)

            _, preds=torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / len(dataloaders['test'].dataset)

    print('Acc: {:.4f}'.format(acc))

print("[Training the model begun ....]")
model_after_train, acc_history = train_model(model_conv, dataloaders, criterion, optimizer_conv, epochs, False)
print("\n[Test model begun ....]")
test_model(model_after_train, test_dataloader)
