# Import packages
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import random
import cv2


# Reading the images
oa_moderate = []
oa_healthy = []


def data(x,y):
    path = x
    for i in glob.iglob(path):
        img = cv2.imread(i)
        # images already (128, 128), do it by convention
        img = cv2.resize(img, (128, 128))
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        y.append(img)

data('./dataset/Healthy/*.jpg', oa_healthy)
data('./dataset/Moderate/*.jpg', oa_moderate)  

oa_moderate = np.array(oa_moderate)
oa_healthy = np.array(oa_healthy)
All = np.concatenate((oa_moderate,oa_healthy))

 #print(np.random.choice (7, 6, replace = False))

#Visualizing OA MRI images
#plt.imshow(oa_severe[0])
#plt.show()

def plot_random(oa_healthy, oa_moderate, num=5):
    oa_healthy_imgs = oa_healthy[(np.random.choice (oa_healthy.shape[0], num, replace = False))]
    oa_moderate_imgs = oa_moderate[(np.random.choice (oa_moderate.shape[0], num, replace = False))]

    plt.figure(figsize= (16,9))
    for f in range(num):
        plt.subplot(1, num, f+1)
        plt.title('healthy')
        plt.imshow(oa_healthy_imgs[f])

    plt.figure(figsize= (16,9))
    for f in range(num):
        plt.subplot(1, num, f+1)
        plt.title('osteoarthritis')
        plt.imshow(oa_moderate_imgs[f])
        

plot_random(oa_healthy, oa_moderate)
#plt.show()

#Creating torch datasets

class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
    def __add__(self,other):
        return ConcatDataset([self, other])
    
class MRI(Dataset):
    def __init__(self):
        # Prepare images
        oa_moderate = []
        oa_healthy = []
        


        for i in glob.iglob('./dataset/Healthy/*.jpg'):
            img = cv2.imread(i)
            # Resize images to (128, 128)
            img = cv2.resize(img, (128, 128))
            # Convert BGR to RGB
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            oa_healthy.append(img)

        for i in glob.iglob('./dataset/Moderate/*.jpg'):
            img = cv2.imread(i)
            # Resize images to (128, 128)
            img = cv2.resize(img, (128, 128))
            # Convert BGR to RGB
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            oa_moderate.append(img)

     

        # Convert lists to numpy arrays
        oa_moderate = np.array(oa_moderate, dtype=np.float32)
        oa_healthy = np.array(oa_healthy, dtype=np.float32)

        # Prepare labels
        label_mapping = {
            'Moderate': 0,
            'Healthy': 1,
        }

        labels_healthy = np.full(len(oa_healthy), label_mapping['Healthy'], dtype=np.int64)
        labels_moderate = np.full(len(oa_moderate), label_mapping['Moderate'], dtype=np.int64)

        # Concatenate all images and labels
        self.images = np.concatenate((oa_healthy,oa_moderate), axis=0)
        self.labels = np.concatenate((labels_healthy, labels_moderate))

        # Normalize images
    def normalize(self):
        self.images = self.images/255.0  # Normalize pixel values to [0, 1]

        # Assign to instance variables

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        sample = {'image': self.images[index], 'label': self.labels[index]}
        return sample
mri_dataset = MRI()
mri_dataset.normalize()


"""index = list(range(len(mri)))
random.shuffle(index)
for idx in index:
    sample = mri[idx]
    img = sample['image']
    label = sample['label']
    plt.title(label)
    plt.imshow(img)
    plt.show()

it = iter(mri)
for i in range(5):
    sample = next(it)
    img = sample['image']
    label = sample['label']
    plt.title(label)
    plt.imshow(img)
    plt.show()
"""

 # creating a dataloader
dataloader = DataLoader(mri_dataset, batch_size = 10, shuffle=True)
for sample in dataloader:
    img = sample['image']
   # plt.imshow(img)
   # plt.show()


#Create a model
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=5),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=5) )

        self.fc_model = nn.Sequential(
        nn.Linear(in_features=256, out_features=120),
        nn.Tanh(),
        nn.Linear(in_features=120, out_features=84),
        nn.Tanh(),
        nn.Linear(in_features=84, out_features=1))

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        x = F.sigmoid(x)
        return x
        
model = CNN()
#print(model.cnn_model[0].weight.shape)

#Linear layers
model.fc_model 

#Evaluate a New-born neural network
mri_dataset = MRI()
mri_dataset.normalize()
device = torch.device('cuda:0')
model = CNN().to(device)

dataloader = DataLoader(mri_dataset, batch_size=32, shuffle=False)

model.eval()
outputs =[]
y_true = []

with torch.no_grad():

    for D in dataloader:
        image = D['image'].to(device)
        label = D['label'].to(device)
        
        y_hat = model(image)
        outputs.append(y_hat.cpu().detach().numpy())
        y_true.append(label.cpu().detach().numpy())

        outputs = np.concatenate(outputs, axis=0).squeeze()
        y_true = np.concatenate(y_true, axis=0).squeeze()

        def threshold(scores, threshold=0.50, minimum=0, maximum=1.0):
            x = np.array(list(scores))
            x[x >= threshold] = maximum
            x[x < threshold] = minimum
            return x
        print(accuracy_score(y_true, threshold(outputs)))

