import numpy as np
import tensorflow as tf
import functions
import os
from tensorflow_probability.python.layers import Convolution2DFlipout, DenseFlipout

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

# Shuffle the data
indices = np.random.permutation(num_samples)

x_data = x_data[indices]
y_data = y_data[indices]

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate the number of samples for each set
num_train_samples = int(train_ratio * num_samples)
num_val_samples = int(val_ratio * num_samples)
num_test_samples = int(test_ratio * num_samples)


x_data = tf.expand_dims(x_data,axis=3,name='channels')
# Split the data into train, validation, and test sets
x_train = x_data[:num_train_samples]
y_train = y_data[:num_train_samples]
x_val = x_data[num_train_samples:num_train_samples+num_val_samples]
y_val = y_data[num_train_samples:num_train_samples+num_val_samples]
x_test = x_data[num_train_samples+num_val_samples:]
y_test = y_data[num_train_samples+num_val_samples:]
"""
# Verify the shapes of the sets
print("Train set shape:", x_train.shape, y_train.shape)
print("Validation set shape:", x_val.shape, y_val.shape)
print("Test set shape:", x_test.shape, y_test.shape)

training_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

# splitta il dataset in train,validation, e test set
train_size = int(0.8 * len(x_data))
val_size = int(0.1 * len(x_data))
test_size = len(x_data) - train_size - val_size

# 80% dei dati è per il training, 10% per il validation set, 10% per il test set
training_dataset = training_dataset.shuffle(len(training_dataset)).batch(16).take(train_size)
val_dataset = training_dataset.skip(train_size).take(val_size)
test_dataset = training_dataset.skip(train_size + val_size).take(test_size) 
"""
model = tf.keras.Sequential([
    Convolution2DFlipout(32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    Convolution2DFlipout(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    Convolution2DFlipout(128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    Convolution2DFlipout(256, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    DenseFlipout(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    DenseFlipout(8,activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics="accuracy")

history = model.fit(x_train,y_train, epochs=100, validation_data=(x_val,y_val), batch_size=16)
# You can assess the model's performance using metrics such as accuracy,
# precision, recall, and F1-score on a separate validation or test set.
model.save('100LR0001.h5py')
# model.evaluate(x_test,y_test)

# model.predict(x_test)