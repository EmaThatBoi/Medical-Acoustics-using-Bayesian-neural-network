import numpy as np
import tensorflow as tf
import functions
import os
from tensorflow import keras
from tensorflow_probability.python.layers import Convolution2DFlipout, DenseFlipout
from tensorflow.python.keras.layers import MaxPooling2D, Flatten, Dropout

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
x_data = tf.expand_dims(x_data,axis=3,name='channels')
training_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

# splitta il dataset in train,validation, e test set
train_size = int(0.8 * len(x_data))
val_size = int(0.1 * len(x_data))
test_size = len(x_data) - train_size - val_size

# 80% dei dati è per il training, 10% per il validation set, 10% per il test set
training_dataset = training_dataset.shuffle(len(training_dataset)).batch(8).take(train_size)
val_dataset = training_dataset.skip(train_size).take(val_size)
test_dataset = training_dataset.skip(train_size + val_size).take(test_size)

model = keras.Sequential()
model.add(Convolution2DFlipout(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2DFlipout(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2DFlipout(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(DenseFlipout(256, activation='relu'))
model.add(Dropout(0.5))
model.add(DenseFlipout(8, activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=opt,
              loss="categorical_crossentropy",
              metrics="accuracy")

history = model.fit(training_dataset, epochs=20, validation_data=val_dataset)

model.save('LR0005.h5py')
# model.evaluate(test_dataset)
