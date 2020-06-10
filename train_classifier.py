from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.model_selection import train_test_split
import argparse
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--datadir', type=str)
args = parser.parse_args()

#CHECK FOR GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#GENERAL PARAMETERS
#   to the ImageFolder structure
data_dir = args.datadir

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "vgg"

# Number of classes in the dataset
num_classes = 9

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 50

input_size = 224
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#define sets
# Create training and validation datasets
#TODO create ONE Dataset

#TODO Split in TWO subsets
train_set = datasets.ImageFolder(data_dir, data_transforms['train'])
val_set = datasets.ImageFolder(data_dir, data_transforms['val'])
print("train_set", train_set.classes)
print("val_set", val_set.classes)

datset_size = len(train_set)
indices = np.arange(datset_size)

train_size = int(0.8 * datset_size)

train_ind = np.random.choice(indices, size=train_size, replace=False)
test_ind = np.setdiff1d(indices, train_ind)

train_set = torch.utils.data.Subset(train_set, train_ind)
val_set = torch.utils.data.Subset(val_set, test_ind)
image_datasets = {x: dset for dset, x in zip([train_set, val_set], ['train', 'val'])}

# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# define classes

model = models.vgg16(pretrained=True)
#print(model)
# (fc): Linear(in_features=2048, out_features=1000, bias=True)
model.classifier[6] = nn.Linear(4096, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#TRANSFORM TO CUDA
model.to(device)

#for name,param in model.named_parameters():
#    if param.requires_grad == True:
#        print("\t",name)

#TRAINING
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
            running_corrects = 0

            # Iterate over data.
            for it, (inputs, labels) in enumerate(dataloaders[phase]):
                #print('Iter {}/{}'.format(it, len(dataloaders[phase].dataset)))
                inputs = inputs.to(device)
                labels = labels.to(device)

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

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, is_inception=False)

#TODO: save model
PATH = './FMGAN_net.pth'
torch.save(model.state_dict(), PATH)
