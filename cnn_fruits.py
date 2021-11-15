# -*- coding: utf-8 -*-
"""CNN_Fruits.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HcVxmMdxMHrYnrf0BojlQJod3vogei3K
"""

from google.colab import drive
drive.mount("/content/gdrive")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

img = image.load_img("/content/gdrive/MyDrive/kaggle_dataset/fruits/fruits-360_dataset/fruits-360/Training/Blueberry/0_100.jpg")
plt.imshow(img)

train = ImageDataGenerator(
    rescale = 1/255,
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7,1.4],
    horizontal_flip=True,
    vertical_flip=True)
validation = ImageDataGenerator(
    rescale = 1/255,
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7,1.4],
    horizontal_flip=True,
    vertical_flip=True)

train_dataset = train.flow_from_directory("/content/gdrive/MyDrive/kaggle_dataset/fruits/fruits-360_dataset/fruits-360/Training",
                                               target_size = (100,100),
                                               batch_size = 10)
validation_dataset = validation.flow_from_directory("/content/gdrive/MyDrive/kaggle_dataset/fruits/fruits-360_dataset/fruits-360/Test", 
                                              target_size = (100,100),
                                              batch_size = 10)

train_dataset.class_indices
validation_dataset.class_indices

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(100,100,3)),
  tf.keras.layers.MaxPool2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
  tf.keras.layers.MaxPool2D(2,2),
  tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
  tf.keras.layers.MaxPool2D(2,2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss='mean_squared_error',
              optimizer = 'adam',
              metrics = ['accuracy'])

from tensorflow.keras.callbacks import TensorBoard
tensorboardCNN = TensorBoard(log_dir="logs/cnn")
model_fit = model.fit(train_dataset,
                      steps_per_epoch = 3,
                      epochs = 5,
                      validation_data = validation_dataset,
                      callbacks=[tensorboardCNN])

import gc
gc.collect()

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext tensorboard
# %tensorboard --logdir logs