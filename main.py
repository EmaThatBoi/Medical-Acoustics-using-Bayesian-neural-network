import numpy as np
import sklearn.dummy
import tensorflow as tf
import tensorflow_probability as tfp
import functions
import os
from tensorflow import keras
from keras.layers import MaxPooling2D,Dropout,Flatten, BatchNormalization
from keras.models import Sequential
from tensorflow_probability.python.layers import Convolution2DFlipout, DenseFlipout
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Model

# script pre-processing.py -> spettri suddivisi per diagnosi nelle corrispondenti directory
#                             ndarray paddati con 0.
"""
dir_path = 'diagnosi/'
dir_label = os.listdir('diagnosi/')

# estrae le matrici degli spettrogrammi dalle directory
data = functions.create_data(dir_path, dir_label)

#standardizza il dataset (media vicina a 0 e std a 1 su tutto il dataset )
media = np.array([np.mean(arr) for arr,_ in data])
std = np.array([np.std(arr,ddof=1) for arr, _ in data])

for i in range(len(data)):
    arr, label = data[i]
    matrice_standard = ((arr - media[i]) / std[i])
    data[i] = (matrice_standard, label)

# crea la coppia - (spettrogramma, label corrispondente)

x_data, y_data = functions.prepare_data(data, dir_label)

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

np.save(file='dataset',arr=x_data)
np.save(file='labels',arr=y_data)

per vedere la distribuzione della lunghezza degli spettrogrammi
leng = [len(x_data[i][0]) for i in range(len(x_data))] #lunghezza spettrogrammi
plt.hist(leng, bins='auto')  # 'auto' automatically determines the number of bins
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.xlim(0,100)
plt.title('Distribution of Matrix Lengths')
plt.show()

"""
x_data,y_data = np.load('dataset.npy'),np.load('labels.npy')

#undersampling
x_data = x_data.reshape(x_data.shape[0], -1)  # shape diventa (920, 128*1723)
ratio = {0: 6, 1: 6, 2: 6, 3: 6, 4: 6, 5: 6, 6: 6, 7: 6}
rus = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
x_data, y_data = rus.fit_resample(x_data, y_data)
x_data = x_data.reshape(x_data.shape[0], 128, 500)

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
# Splitta il dataset in train, validation, and test set
x_train = x_data[:num_train_samples]
y_train = y_data[:num_train_samples]
#validation set
x_val = x_data[num_train_samples:num_train_samples+num_val_samples]
y_val = y_data[num_train_samples:num_train_samples+num_val_samples]
# test set
x_test = x_data[num_train_samples+num_val_samples:]
y_test = y_data[num_train_samples+num_val_samples:]



model = Sequential([
    Convolution2DFlipout(16, kernel_size=(5,5), activation='relu',
                         activity_regularizer=None,seed=1 ),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Convolution2DFlipout(32, kernel_size=3, activation='relu',
                         activity_regularizer=None,seed=2),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Convolution2DFlipout(64, kernel_size=3, activation='relu',
                         activity_regularizer=None,seed=3),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    DenseFlipout(256, activation='relu', seed=1),
    BatchNormalization(),
    DenseFlipout(16, activation='relu', seed=2),
    BatchNormalization(),
    #Dropout(0.2),
    DenseFlipout(8, activation='softmax', seed=3)
])

opt = tf.keras.optimizers.Adagrad(learning_rate=2.0, weight_decay=0.002)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=2, validation_data=(x_val,y_val), batch_size=8)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()

#model.save('LR002.h5py')
#model.evaluate(test_dataset)

### per mandare in overfit, provare con pochi dati, loss bassa ma val-loss alta

# plottare loss
# plottare heatmap dati

# aumentare lr,
