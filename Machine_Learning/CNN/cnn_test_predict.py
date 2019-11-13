import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from PIL import Image

plt.close("all")

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', \
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class_names = ['0', '1', '2', '3', '4', \
               '5', '6', '7', '8', '9']


model = keras.models.load_model("numerical_mnist_adam_categorical10_cnn.h5")

test_image_string='test_andy_2_thick.jpg'
test_image = Image.open(test_image_string)
test_image = test_image.resize((28,28))
plt.imshow(test_image)
plt.show()
test_image = np.array(test_image)
test_image = test_image/255.0
test_image = np.resize(test_image,(1,1,28,28))
prediction = model.predict_classes(test_image)
prediction_class = class_names[prediction[0]]
print(f"\nI predict that {test_image_string} is a {prediction_class} !")
