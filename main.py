# Import packages
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
        plt.imshow(oa_healthy_imgs[f])

    plt.figure(figsize= (16,9))
    for f in range(num):
        plt.subplot(1, num, f+1)
        plt.title('severe')
        plt.imshow(oa_severe_imgs[f])
        

plot_random(oa_healthy, oa_severe, oa_doubtful, oa_minimal, oa_moderate)
plt.show()

#Creating torch datasets