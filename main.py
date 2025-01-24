# Import packages
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import cv2


# Reading the images
oa_severe = []
oa_moderate = []
oa_minimal = []
oa_healthy = []
oa_doubtful = []

def data(x,y):
    path = x
    for i in glob.iglob(path):
        img = cv2.imread(i)
        # images already (128, 128), do it by convention
        img = cv2.resize(img, (128, 128))
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        y.append(img)



data('./dataset/Doubtful/*.jpg', oa_doubtful)
data('./dataset/Healthy/*.jpg', oa_healthy)
data('./dataset/Minimal/*.jpg', oa_minimal)
data('./dataset/Moderate/*.jpg', oa_moderate)  
data('./dataset/Severe/*.jpg', oa_severe)

oa_severe = np.array(oa_severe)
oa_moderate = np.array(oa_moderate)
oa_minimal = np.array(oa_minimal)
oa_healthy = np.array(oa_healthy)
oa_doubtful = np.array(oa_doubtful)
All = np.concatenate((oa_severe, oa_moderate, oa_minimal, oa_healthy, oa_doubtful))

 #print(np.random.choice (7, 6, replace = False))

#Visualizing OA MRI images
#plt.imshow(oa_severe[0])
#plt.show()

def plot_random(oa_healthy, oa_severe, oa_doubtful, oa_minimal, oa_moderate, num=5):
    oa_healthy_imgs = oa_healthy[(np.random.choice (oa_healthy.shape[0], num, replace = False))]
    oa_severe_imgs = oa_severe[(np.random.choice (oa_severe.shape[0], num, replace = False))]
    oa_doubtful_imgs = oa_doubtful[(np.random.choice (oa_doubtful.shape[0], num, replace = False))]
    oa_minimal_imgs = oa_minimal[(np.random.choice (oa_minimal.shape[0], num, replace = False))]
    oa_moderate_imgs = oa_moderate[(np.random.choice (oa_moderate.shape[0], num, replace = False))]

    plt.figure(figsize= (16,9))
    for f in range(num):
        plt.subplot(1, num, f+1)
        plt.title('healthy')
        plt.imshow(oa_doubtful_imgs[f])

    plt.figure(figsize= (16,9))
    for f in range(num):
        plt.subplot(1, num, f+1)
        plt.title('severe')
        plt.imshow(oa_severe_imgs[f])
        

plot_random(oa_healthy, oa_severe, oa_doubtful, oa_minimal, oa_moderate)
plt.show()

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
        oa_severe = []
        oa_moderate = []
        oa_minimal = []
        oa_healthy = []
        oa_doubtful = []

        def data(x, y):
            path = x
            for i in glob.iglob(path):
                img = cv2.imread(i)
                # Resize images to (128, 128)
                img = cv2.resize(img, (128, 128))
                # Convert BGR to RGB
                b, g, r = cv2.split(img)
                img = cv2.merge([r, g, b])
                y.append(img)

        # Load images into respective lists
        data('./dataset/Doubtful/*.jpg', oa_doubtful)
        data('./dataset/Healthy/*.jpg', oa_healthy)
        data('./dataset/Minimal/*.jpg', oa_minimal)
        data('./dataset/Moderate/*.jpg', oa_moderate)
        data('./dataset/Severe/*.jpg', oa_severe)

        # Convert lists to numpy arrays
        oa_severe = np.array(oa_severe, dtype=np.float32)
        oa_moderate = np.array(oa_moderate, dtype=np.float32)
        oa_minimal = np.array(oa_minimal, dtype=np.float32)
        oa_healthy = np.array(oa_healthy, dtype=np.float32)
        oa_doubtful = np.array(oa_doubtful, dtype=np.float32)

        # Prepare labels
        label_mapping = {
            'Doubtful': 0,
            'Healthy': 1,
            'Minimal': 2,
            'Moderate': 3,
            'Severe': 4
        }

        labels_doubtful = np.full(len(oa_doubtful), label_mapping['Doubtful'], dtype=np.int64)
        labels_healthy = np.full(len(oa_healthy), label_mapping['Healthy'], dtype=np.int64)
        labels_minimal = np.full(len(oa_minimal), label_mapping['Minimal'], dtype=np.int64)
        labels_moderate = np.full(len(oa_moderate), label_mapping['Moderate'], dtype=np.int64)
        labels_severe = np.full(len(oa_severe), label_mapping['Severe'], dtype=np.int64)

        # Concatenate all images and labels
        all_images = np.concatenate((oa_doubtful, oa_healthy, oa_minimal, oa_moderate, oa_severe))
        all_labels = np.concatenate((labels_doubtful, labels_healthy, labels_minimal, labels_moderate, labels_severe))

        # Normalize images
        all_images = all_images / 255.0  # Normalize pixel values to [0, 1]

        # Assign to instance variables
        self.images = all_images
        self.labels = all_labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        sample = {'image': self.images[index], 'label': self.labels[index]}
        return sample
mri = MRI()
print(mri[5])
 # creating a dataloader
dataloader = DataLoader(mri)
for sample in dataloader:
    img = sample['image']


