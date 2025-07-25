#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import LogNorm 
from io import BytesIO
import base64
import extcolors
import os
import shutil

# yolo import
import pandas
import torch
import glob

# mongo query
from rembg import remove
import xlwings as xw
import boto3
from io import BytesIO
import pymongo
import sys

# cnn import
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set the path for the trained model
modelPath = 'D:\\FaceDetection\\model\\'

# Load the VGG16 model with pre-trained weights and without the top (classification) layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Create a new sequential model and add the pre-trained VGG16 base
model = Sequential()
model.add(base_model)

# Add custom layers for classification on top of the VGG16 base
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Data preprocessing and training remains the same
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory(
    r'D:\OneDrive - CONTEXIO LLP\Backup before formatting\computer-vision\NonCroppedSleeves\sleevedataFS\trainingsmallsamples',
    target_size=(200, 200),
    batch_size=32
)

validation_dataset = validation.flow_from_directory(
    r'D:\OneDrive - CONTEXIO LLP\Backup before formatting\computer-vision\NonCroppedSleeves\sleevedataFS\validationsmallsamples',
    target_size=(200, 200),
    batch_size=32
)

# Training the model
model_fit = model.fit(
    train_dataset,
    steps_per_epoch=50,
    epochs=5,
    validation_data=validation_dataset
)

# Create the directory if it doesn't exist
if not os.path.exists(modelPath):
    os.makedirs(modelPath)

# Save the trained model to the specified directory
model.save(os.path.join(modelPath, "sleeve_trained_model_FS_VGG16.h5"))


# In[2]:


# Save the trained model to the specified directory
model.save(os.path.join(modelPath, "sleeve_trained_model_FS_VGG16.h5"))


# In[3]:


#Making Python Script 
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.colors import LogNorm 
from io import BytesIO
import base64
import extcolors
import shutil
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Replace 'your_model.h5' with the actual path to your trained model
model = load_model('D:\\FaceDetection\\model\\sleeve_trained_model_FS_VGG16.h5')

dir_path = "D:/OneDrive - CONTEXIO LLP/Backup before formatting/computer-vision/testing/"

new_size = (200, 200)
for i in os.listdir(dir_path):
    img = Image.open(os.path.join(dir_path, i))   #Use os.path.join for path
    img = img.resize(new_size)
    plt.imshow(img)
    plt.show()

    X = img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    predicted_class = np.argmax(val)
    print(predicted_class)
    print(val)
    if predicted_class == 1:
        print('Full Sleeves')
    elif predicted_class == 2:
        print('Half Sleeves')
    elif predicted_class == 3:
        print('Sleeveless')
    elif predicted_class == 0:
        print('3/4th Sleeves')
    else:
        print("n/a")


# In[5]:


from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import shutil
import os

dir_path = "D:/OneDrive - CONTEXIO LLP/Backup before formatting/computer-vision/testing/"
output_folder = "D:/OneDrive - CONTEXIO LLP/Backup before formatting/computer-vision/output"
new_size = (200, 200)  # Change this line

model = load_model('D:\\FaceDetection\\model\\sleeve_trained_model_FS_VGG16.h5')

for i in os.listdir(dir_path):
    img_path = os.path.join(dir_path, i)
    img = Image.open(img_path)
    img = img.resize(new_size)
    plt.imshow(img)
    plt.show()

    X = img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    predicted_class = np.argmax(val)
    print(predicted_class)
    print(val)

    if predicted_class == 1:
        print('Full Sleeves')
        output_path = os.path.join(output_folder, 'full_sleeves', i)
    elif predicted_class == 2:
        print('Half Sleeves')
        output_path = os.path.join(output_folder, 'half_sleeves', i)
    elif predicted_class == 3:
        print('Sleeveless')
        output_path = os.path.join(output_folder, 'sleeveless', i)
    elif predicted_class == 0:
        print('3/4th Sleeves')
        output_path = os.path.join(output_folder, 'three_fourth_sleeves', i)
    else:
        print("n/a")
        continue  # Skip saving if class is not recognized

    shutil.copy(img_path, output_path)
    print(f"Image saved to {output_path}")


# In[ ]:




