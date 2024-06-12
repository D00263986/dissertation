import pandas as pd
import numpy as np
import itertools as it
import sys
import os
import matplotlib.pyplot as plt
import PIL.Image as pil_image
import time

import tensorflow as tf
from tensorflow.keras import layers as klayers
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Input, 
                                     BatchNormalization, Dropout)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from numpy import expand_dims

train_augmentor = IDG(rescale=1./255,
                      rotation_range=30,
                      width_shift_range=0.2, 
                      height_shift_range=0.2,
                      horizontal_flip=True, 
                      vertical_flip=True,
                      brightness_range=[0.1, 1],
                      zoom_range=0.3
                     )

val_augmentor = IDG(rescale=1./255)
test_augmentor = IDG(rescale=1./255)

train_data = train_augmentor.flow_from_directory("data/train/", class_mode="categorical", 
                                                 shuffle=False, batch_size=100, target_size=(299, 299))
val_data = val_augmentor.flow_from_directory("data/val/", class_mode="categorical", 
                                             shuffle=False, batch_size=100, target_size=(299, 299))
test_data = test_augmentor.flow_from_directory("data/test/", class_mode="categorical", 
                                               shuffle=False, batch_size=50, target_size=(299, 299))

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))

base_model.summary()

for layer in base_model.layers:
    layer.trainable = False

custom_model = Sequential([
    base_model,
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

custom_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

custom_model.summary()

start_time = time.time()
callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.0001),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
]

history = custom_model.fit(train_data, epochs=20, callbacks=callbacks_list, validation_data=val_data)

print("Model training time: ", int(time.time() - start_time), 's')

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

loaded_model = load_model('model_new.h5')

results = loaded_model.evaluate(test_data, workers=1)
