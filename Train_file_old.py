# Imports here
import torch
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as TF
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import json
import argparse

#get our command line arguments
def get_input_args():
    parser = argparse.ArgumentParser(description='Process values for arguments')
    # Create 2 command line arguments: one for the flower image directory, the other for the architecture
    parser.add_argument('--dir', type = str, default = 'flowers/', help='Path to folder of images')
    parser.add_argument('--arch', type = str, default = 'vgg', help = "Architecture for the NN model")
    parser.add_argument('--hidden_units', type = int, default = 512, help = "Number of hidden training units")
    parser.add_argument('--learnrate', type = float, default = 0.01, help = "Scalar to multiply loss by")
    parser.add_argument('--epochs', type = int, default = 20, help = "Number of rounds of training")
    parser.add_argument('--flower_cat', type = str, default = 'cat_to_name.json', help = "file with flower categories")
    return parser.parse_args()

args = get_input_args()

print("dir: ", args.dir)
print("arch: ", args.arch)
print("hidden units: ", args.hidden_units)
print("learnrate: ", args.learnrate)
print("epochs: ", args.epochs)
print("flower categories file: ", args.flower_cat)

print("Setting up directories, transforms, and dataloaders...")
data_dir = 'flowers'#args.dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Dataset transforms
train_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.485, 0.485), (0.229, 0.224, 0.225))])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#Datasets
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(train_dir, transform=valid_transforms)

#Dataloaders
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=24, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=24, shuffle=True)

#Make category to name dictionary
print("setting up dictionary converter")
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
#print(cat_to_name)

#Get the pre-trained neural network
print("Setting up device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Setting up models")
model = models.densenet121(pretrained = True)#models.vgg11(pretrained = True)
print("MODEL dimensions: ")
print(model)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

print("Setting up classifier")
'''model.classifier = nn.Sequential(nn.Linear(25088, 2048),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 102),
                                 nn.LogSoftmax(dim=1))'''
model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 102),
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device);

print(model)

# TODO: Do validation on the test set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_losses, test_losses = [], []
epochs = 3
steps = 0


# Turn off gradients for validation, saves memory and computations

print("Training and Testing Model ", epochs, " times")
for e in range(epochs):
    running_loss = 0
    print("Training...")
    for images, labels in train_dataloader:
        steps += 1
        # Move input and label tensors to the default device
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(images)
        loss = criterion(logps, labels)
        loss.backward()
        print("Current Loss: ", loss)
        optimizer.step()
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0
        print("Testing...")
        ## TODO: Implement the validation pass and print out the validation accuracy
        with torch.no_grad():
            model.eval()
            for images, labels in valid_dataloader:
                images, labels = images.to(device), labels.to(device)
                log_ps = model.forward(images)
                test_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        model.train()
        train_losses.append(running_loss/len(train_dataloader))
        test_losses.append(test_loss/len(valid_dataloader))
        print("Epoch:  {}/{}.. ".format(e+1, epochs),
              "Train Loss: {:.3f}.. ".format(running_loss/len(train_dataloader)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(valid_dataloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(valid_dataloader)))

#Save the checkpoint
print("Saving checkpoint file")
checkpoint = {'input_size': 25088,
              'output_size': 102,
              'epochs': 2,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')
print("Finished Training file")