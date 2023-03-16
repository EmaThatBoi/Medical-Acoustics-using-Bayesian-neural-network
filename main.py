import numpy as np
import tensorflow as tf
from tensorflow_probability.python.layers import Convolution2DFlipout, DenseFlipout
from tensorflow.python.keras.layers import MaxPooling2D, Flatten, Dropout
from tensorflow.python.keras import Sequential
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import functions
import os

# pre-processing.py -> spettri suddivisi per diagnosi nelle corrispondenti directory

dir_path = 'diagnosi/'
dir_label = os.listdir('diagnosi/')

data = functions.create_data(dir_path, dir_label)

x_data, y_data = functions.prepare_data(data, dir_label)

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

indices = np.arange(len(x_data))
np.random.shuffle(indices)
x_data = x_data[indices]
y_data = y_data[indices]

# Split the data into train, validation, and test sets
train_size = int(0.8 * len(x_data))
val_size = int(0.1 * len(x_data))
test_size = len(x_data) - train_size - val_size

x_train, y_train = x_data[:train_size], y_data[:train_size]
x_val, y_val = x_data[train_size:train_size+val_size], y_data[train_size:train_size+val_size]
x_test, y_test = x_data[train_size+val_size:], y_data[train_size+val_size:]

# Fit a dummy classifier and make predictions
dummy_clf = DummyClassifier(strategy="stratified")
dummy_clf.fit(x_train, y_train)
y_pred = dummy_clf.predict(x_test)

# Evaluate the accuracy of the dummy classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the dummy classifier: {accuracy}")