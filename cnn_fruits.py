from google.colab import drive
drive.mount("/content/gdrive")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

train_dataset = tf.keras.preprocessing.image_dataset_from_directory("/content/gdrive/MyDrive/Datasets/archive/train",
                                                               image_size=(224, 224),
                                                              batch_size=32,
                                                              label_mode="categorical")
val_dataset = tf.keras.preprocessing.image_dataset_from_directory("/content/gdrive/MyDrive/Datasets/archive/validation",
                                                              image_size=(224, 224),
                                                              batch_size=32,
                                                              label_mode="categorical")

from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from tensorflow.keras.layers import (BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(len(train_dataset.class_names),activation='softmax'))

model.compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = ["accuracy"])
model.summary()

fitted_model = model.fit(train_dataset,
                    epochs= 15,
                    validation_data=val_dataset)

plt.plot(fitted_model.history['loss'])
plt.plot(fitted_model.history['val_loss'])
plt.title("Evolución de la función de coste")
plt.ylabel("Coste")
plt.xlabel('Épocas')
plt.legend(["coste (train)", "coste (val)"])
plt.show()

plt.plot(fitted_model.history['accuracy'])
plt.plot(fitted_model.history['val_accuracy'])
plt.title("Evolución de la precisión")
plt.ylabel("Accuracy")
plt.xlabel('Épocas')
plt.legend(['precisión (train)', 'precision (val)'])
plt.show()
