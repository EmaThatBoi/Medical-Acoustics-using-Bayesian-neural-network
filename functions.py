import numpy as np
import os
import tensorflow as tf

def create_data(dir_path,dir_label):
    """

    :param dir_path: path della directory contenente le immagini separate per diagnosi.
    :param dir_label: array di cartelle separate per diagnosi.
    :return: array con immagini e corrispondente label della diagnosi.
    """
    training_data = []
    for label, category in enumerate(dir_label):
        path = os.path.join(dir_path, category)
        for filename in os.listdir(path):       # ogni file nei path, viene aperto e preparato
                                                 # in una lista sarà data in input
            filepath = os.path.join(path,filename)
            #img = tf.io.read_file(filepath)
            #img = tf.io.decode_image(img, channels=1)
            img = np.load(filepath)
            training_data.append((img,label))

    return training_data

def prepare_data(training_data,dir_label):
    """

    :param data: prende in input una struttura dati contenente immagine e label
    :return: x - dati di training e y - label corrispondenti
    """
    x_train = []
    y_train = []
    for image,label in training_data:
        x_train.append(image)
        y = np.zeros(len(dir_label))
        y[label] = 1              # assegna 1 alla classe di appartenenza, che viene indicata da label
        y_train.append(y)

    return x_train, y_train