import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

class_names = ['0', '1', '2', '3', '4', \
               '5', '6', '7', '8', '9']

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', \
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images.shape

train_images = np.expand_dims(train_images,axis=1)
test_images = np.expand_dims(test_images,axis=1)

model = keras.Sequential([ \
    keras.layers.Conv2D(filters=30,kernel_size=2,strides=(1, 1),activation='relu',data_format = "channels_first"),\
    keras.layers.MaxPooling2D(pool_size=(2, 2),data_format= "channels_first"),\
    keras.layers.Flatten(),\
    keras.layers.Dense(1470, activation='relu'),\
    keras.layers.Dense(735, activation='relu'),\
    keras.layers.Dense(10, activation='softmax')\
])

model.compile(optimizer='adam',\
              loss='sparse_categorical_crossentropy',\
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=3)
model.save("numerical_mnist_adam_categorical10_cnn.h5")
model.evaluate(test_images, test_labels, verbose=1)
