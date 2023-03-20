import numpy as np
import tensorflow as tf
import functions
import os
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

# Get the number of samples
num_samples = x_data.shape[0]

# Shuffle dei dati
indices = np.random.permutation(num_samples)
x_data = x_data[indices]
y_data = y_data[indices]

# numero di campioni per ogni set
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
num_train_samples = int(train_ratio * num_samples)
num_val_samples = int(val_ratio * num_samples)
num_test_samples = int(test_ratio * num_samples)

x_data = tf.expand_dims(x_data,axis=3,name='channels')

# Splitta il dataset in train, validation, e test sets
x_train = x_data[:num_train_samples]
y_train = y_data[:num_train_samples]
x_val = x_data[num_train_samples:num_train_samples+num_val_samples]
y_val = y_data[num_train_samples:num_train_samples+num_val_samples]
x_test = x_data[num_train_samples+num_val_samples:]
y_test = y_data[num_train_samples+num_val_samples:]

model = tf.keras.Sequential()
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

# model.build(x_data.shape)
# model.summary()

history = model.fit(x_train,y_train, epochs=20, validation_data=(x_val,y_val))

model.save('LR0005.h5py')

# model.evaluate()
# model.predict(x_test)
