import numpy as np
import pandas as pd
import torch
from torch.nn import Sequential, MaxPool2d, Flatten, ReLU, Sigmoid, BCELoss, Conv2d, Linear, Dropout
import os
import librosa
from librosa import feature
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.nn import Sequential, MaxPool2d, Flatten, ReLU, Sigmoid, BCELoss, Conv2d, Linear, Dropout

def load_dataset():
    pass

def plot_weights(epoch,CNN_model):
    """

    :param epoch: epoch number
    :param model: model entity from which weights values are taken
    :return: distributions plots of the mean and std of the weights
    """
    mean_0 = CNN_model[0].mu_kernel.flatten().cpu().detach().numpy()
    std_0 = CNN_model[0].rho_kernel.flatten().cpu().detach().numpy()
    mean_4 = CNN_model[4].mu_kernel.flatten().cpu().detach().numpy()
    std_4 = CNN_model[4].rho_kernel.flatten().cpu().detach().numpy()
    mean_9 = CNN_model[9].mu_weight.flatten().cpu().detach().numpy()
    std_9 = CNN_model[9].rho_weight.flatten().cpu().detach().numpy()
    mean_12 = CNN_model[12].mu_weight.flatten().cpu().detach().numpy()
    std_12 = CNN_model[12].rho_weight.flatten().cpu().detach().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    fig.suptitle(f'step{epoch}')
    ax1.set_title('Weight means')
    ax2.set_title('weight std')

    sns.kdeplot(mean_0, ax=ax1, shade=True, label='First Conv2Dflipout')
    sns.kdeplot(mean_4, ax=ax1, shade=True, label='Second Conv2Dflipout')
    #sns.kdeplot(mean_9, ax=ax1, shade=True, label='First Linear')
    sns.kdeplot(mean_12, ax=ax1, shade=True, label='Second Linear')

    sns.kdeplot(std_0, ax=ax2, shade=True, label='First Conv2Dflipout')
    sns.kdeplot(std_4, ax=ax2, shade=True, label='Second Conv2Dflipout')
    sns.kdeplot(std_9, ax=ax2, shade=True, label='First Linear')
    sns.kdeplot(std_12, ax=ax2, shade=True, label='Second Linear')

    return plt.show()

def create_CNN_model():
    model = Sequential(
        Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1),
        ReLU(),
        Dropout(0.4),
        MaxPool2d(kernel_size=2, stride=2),
        Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1),
        ReLU(),
        Dropout(0.4),
        MaxPool2d(kernel_size=2, stride=2),
        Flatten(),
        Linear(in_features=32 * 125 * 16, out_features=128),
        ReLU(),
        Dropout(0.4),
        Linear(in_features=128, out_features=1),
        Sigmoid()
    )
    return model

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

def spectrogram(path):
    """
    :param path: percorso della registrazione audio per cui computare il lo spettrogramma di mel
    :return: matrice dello spettrogramma di mel in deciBel
    """
    samples, sr = librosa.load(path, sr=22500)
    signal = feature.melspectrogram(y=samples, sr=sr, hop_length=512)
    signal_dB = librosa.power_to_db(signal)

    #fig = plt.subplot()
    #spectrogram = librosa.display.specshow(signal_dB)

    return signal_dB

def training_set(train_dir_path,train_dir_label):
    # estrae le matrici degli spettrogrammi dalle directory
    train_data = create_data(train_dir_path, train_dir_label)

    # standardizza il dataset (media vicina a 0 e std a 1 su tutto il dataset )
    media = np.array([np.mean(arr) for arr, _ in train_data])
    std = np.array([np.std(arr, ddof=1) for arr, _ in train_data])
    for i in range(len(train_data)):
        arr, label = train_data[i]
        matrice_standard = ((arr - media[i]) / std[i])
        train_data[i] = (matrice_standard, label)

    # crea la coppia - (spettrogramma, label corrispondente)
    x_train_data, y_train_data = prepare_data(train_data, train_dir_label)

    x_train_data = np.array(x_train_data, dtype=np.float32)
    y_train_data = np.array(y_train_data, dtype=np.float32)
    np.save(file='ICBHI set/train_dataset', arr=x_train_data)
    np.save(file='ICBHI set/train_labels', arr=y_train_data)

    return x_train_data,y_train_data

def test_set(test_dir_path,test_dir_label):
    # estrae le matrici degli spettrogrammi dalle directory
    test_data = create_data(test_dir_path, test_dir_label)

    # standardizza il dataset (media vicina a 0 e std a 1 su tutto il dataset)
    media = np.array([np.mean(arr) for arr, _ in test_data])
    std = np.array([np.std(arr, ddof=1) for arr, _ in test_data])

    for i in range(len(test_data)):
        arr, label = test_data[i]
        matrice_standard = ((arr - media[i]) / std[i])
        test_data[i] = (matrice_standard, label)

    # crea la coppia - (spettrogramma, label corrispondente)
    x_test_data, y_test_data = prepare_data(test_data, test_dir_label)

    x_test_data = np.array(x_test_data, dtype=np.float32)
    y_test_data = np.array(y_test_data, dtype=np.float32)
    np.save(file='ICBHI set/test_dataset', arr=x_test_data)
    np.save(file='ICBHI set/test_labels', arr=y_test_data)

    return x_test_data,y_test_data

def balanced_accuracy_per_threshold(high_certainty_predicted,high_certainty_labels):
    high_certainty_predicted_int = high_certainty_predicted.int()
    high_certainty_labels_int = high_certainty_labels.int()
    #label= 0 --> malati, label = 1 --> sani
    TP = (high_certainty_predicted_int & high_certainty_labels_int).sum().item()  # Veri positivi veri sani
    FP = (high_certainty_predicted_int & (1 - high_certainty_labels_int)).sum().item()  # Falsi positivi falsi sani
    TN = ((1 - high_certainty_predicted_int) & (1 - high_certainty_labels_int)).sum().item()  # Veri negativi veri malati
    FN = ((1 - high_certainty_predicted_int) & high_certainty_labels_int).sum().item()  # Falsi negativi falsi malati
    epsilon = 1e-7
    specificity = TN / (TN + FP + epsilon)  # Specificità o tasso di veri negativi
    sensitivity = TP / (TP + FN + epsilon)  # Sensibilità, tasso di rilevamento, recall, o tasso di veri positivi
    balanced_accuracy = (sensitivity + specificity) / 2
    confusion_matrix = pd.DataFrame({
        'Previsione malato': [TN, FN],
        'Previsione sano': [FP, TP]
    }, index=['Vero malato', 'Vero sano'])

    return balanced_accuracy, confusion_matrix


