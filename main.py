import numpy as np
import sklearn.dummy
import tensorflow as tf
import functions
import os
from tensorflow import keras
from keras.layers import MaxPooling2D,Dropout,Flatten
from keras.models import Sequential
from tensorflow_probability.python.layers import Convolution2DFlipout, DenseFlipout
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd

# script pre-processing.py -> spettri suddivisi per diagnosi nelle corrispondenti directory
#                             ndarray paddati con 0.

dir_path = 'diagnosi/'
dir_label = os.listdir('diagnosi/')

# estrae le matrici degli spettrogrammi dalle directory
data = functions.create_data(dir_path, dir_label)

# standardizza i valori delle matrici
flat = np.concatenate([np.ndarray.flatten(arr) for arr, _ in data])
mean = np.mean(flat)
std = np.std(flat)
for i in range(len(data)):
    arr, label = data[i]
    data[i] = ((arr - mean / std), label)

# crea la coppia - (spettrogramma, label corrispondente)

x_data, y_data = functions.prepare_data(data, dir_label)

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

#undersampling
x_data = x_data.reshape(x_data.shape[0], -1)  # shape diventa (920, 128*1723)
ratio = {0: 1, 1: 3, 2: 3, 3: 3, 4: 3, 5: 2, 6: 3, 7: 3}
rus = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
x_data, y_data = rus.fit_resample(x_data, y_data)
x_data = x_data.reshape(x_data.shape[0], 128, 1723)

# numero dei campioni
num_samples = x_data.shape[0]

# Shuffle dei dati
indices = np.random.permutation(num_samples)

x_data = x_data[indices]
y_data = y_data[indices]

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# numero dei campioni per ogni set
num_train_samples = int(train_ratio * num_samples)
num_val_samples = int(val_ratio * num_samples)
num_test_samples = int(test_ratio * num_samples)

x_data = np.expand_dims(x_data, axis=3)
# Split il dataset in train, validation, and test set
x_train = x_data[:num_train_samples]
y_train = y_data[:num_train_samples]
#validation set
x_val = x_data[num_train_samples:num_train_samples+num_val_samples]
y_val = y_data[num_train_samples:num_train_samples+num_val_samples]
# test set
x_test = x_data[num_train_samples+num_val_samples:]
y_test = y_data[num_train_samples+num_val_samples:]



model = Sequential([
    Convolution2DFlipout(8, kernel_size=15, activation='relu',activity_regularizer=None),
    MaxPooling2D(pool_size=(2,2)),
    Convolution2DFlipout(16, kernel_size=10, activation='relu',activity_regularizer=None),
    MaxPooling2D(pool_size=(2,2)),
    Convolution2DFlipout(32, kernel_size=4, activation='relu',activity_regularizer=None),
    MaxPooling2D(pool_size=(2,2)),
    Convolution2DFlipout(64, kernel_size=3, activation='relu',activity_regularizer=None),
    MaxPooling2D(pool_size=(2,2)),
    Convolution2DFlipout(128, kernel_size=3, activation='relu',activity_regularizer=None),
    MaxPooling2D(pool_size=(2,2)),


    Flatten(),
    DenseFlipout(128, activation='relu'),
    DenseFlipout(256, activation='relu'),
    #Dropout(0.7),
    DenseFlipout(8,activation='softmax')
])

opt = tf.keras.optimizers.Adagrad(learning_rate=1, weight_decay=0.002)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=500, validation_data=(x_val,y_val), batch_size=16)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()

#model.save('LR002.h5py')
#model.evaluate(test_dataset)

### per mandare in overfit, provare con pochi dati, loss bassa ma val-loss alta

# plottare loss
# plottare heatmap dati

# aumentare lr,
