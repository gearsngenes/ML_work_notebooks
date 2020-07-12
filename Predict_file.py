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

data_dir = 'flowers'
train_dir = data_dir + '/train'

train_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.485, 0.485), (0.229, 0.224, 0.225))])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#Load saved model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    '''classifier = nn.Sequential(nn.Linear(25088, 2048),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 102),
                                 nn.LogSoftmax(dim=1))'''
    classifier = nn.Sequential(nn.Linear(1024, 600),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(600, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 102),
                                 nn.LogSoftmax(dim=1))#checkpoint['classifier']
    #model = models.vgg11(pretrained=True)
    model = models.densenet121(pretrained = True)
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    return model
model = load_checkpoint('checkpoint.pth')
print(model)

#processes image to right format
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    e1 = image.size[0]
    e2 = image.size[1]
    if e1 < e2:
        h = int(256/e1 * e2)
        image = image.resize((256, h))
        m = int((h-224)/2)
        image = image.crop((16,m,16+224,m+224))
    else:
        w = int(256/e2 * e1)
        image = image.resize((w, 256))
        m = int((w-224)/2)
        image = image.crop((m,16,m+224,16+224))
    np_image = np.array(image)
    np_image = np.true_divide(np_image, 255)
    np_image -= [0.485, 0.456, 0.406]
    np_image /= [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2,0,1))
    img = torch.FloatTensor([np_image])
    return img

#Show image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.view(image.size()[0]*image.size()[1], image.size()[2], image.size()[3])
    image = image.numpy()
    image = image.transpose((1, 2, 0))
    print(image.shape)
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

#Predict
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = Image.open(image_path)
    img = process_image(img)
    
    log_ps = model.forward(img)
    ps = torch.exp(log_ps)
    probs, classIdxes = ps.topk(topk, dim=1)
    model.class_to_idx = train_data.class_to_idx
    
    idx_to_class = {value : key for key,value in model.class_to_idx.items()}
    
    classes = []
    for n in classIdxes[0]:
        classes.append(cat_to_name[idx_to_class[int(n)]])
    #imshow(img)
    probs = probs.detach().numpy()[0]
    print("Probabilities: ", probs)
    print("Flowers: ", classes)
    plt.subplot(2, 1, 1) # 1 row, 2 cols, subplot 1
    imshow(img, ax=plt.subplot(2,1,1))
    plt.subplot(2, 1, 2) # 1 row, 2 cols, subplot 2
    plt.bar(classes, probs)
    plt.xlabel("flower types")
    plt.ylabel("probability")
    plt.xticks(rotation = 45)

#predict image
img_path = "flowers/test/21/image_06807.jpg"
predict(img_path, model)