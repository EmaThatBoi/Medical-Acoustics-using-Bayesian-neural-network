import numpy as np
import os
import torch
import functions
from torchvision import transforms
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Sequential, MaxPool2d, Flatten, ReLU, Sigmoid, BCELoss, Conv2d, Linear, Dropout, BatchNorm2d,BatchNorm1d
from bayesian_torch.models.dnn_to_bnn import get_kl_loss, dnn_to_bnn
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# script pre-processing.py -> spettri suddivisi per diagnosi nelle corrispondenti directory
#                             ndarray paddati con 0.
"""
x_train, y_train = np.load('ICBHI set/train_dataset.npy'), np.load('ICBHI set/train_labels.npy')
x_test, y_test = np.load('ICBHI set/test_dataset.npy'), np.load('ICBHI set/test_labels.npy')

# len(y_train[y_train[:,0]==0.0]) # numero di esiti sani

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)

# Creiamo il set di validazione dal set di test

x_train = x_train.reshape(x_train.shape[0], -1)  # shape diventa (920, 128*500)
train_ratio = {0: 129, 1: 129} # 0 sani, 1 malati
train_rus = RandomUnderSampler(sampling_strategy=train_ratio, random_state=0)
x_train, y_train = train_rus.fit_resample(x_train, y_train)
x_train = x_train.reshape(x_train.shape[0], 128, 500) #bilanciamento disattivato, 60% train 40% test

np.save(file='official set/train_dataset',arr=x_train)
np.save(file='official set/train_labels',arr=y_train)
np.save(file='official set/test_dataset',arr=x_test)
np.save(file='official set/test_labels',arr=y_test)
np.save(file='official set/val_dataset',arr=x_val)
np.save(file='official set/val_labels',arr=y_val)"""

x_train, y_train = np.load('official set/train_dataset.npy'), np.load('official set/train_labels.npy')
x_test, y_test = np.load('official set/test_dataset.npy'), np.load('official set/test_labels.npy')
x_val,y_val = np.load('official set/val_dataset.npy'), np.load('official set/val_labels.npy')

# Aggiungiamo dimensione del canale ai dati

x_train = x_train.reshape(-1, 1, 128, 500)
x_test = x_test.reshape(-1, 1, 128, 500)
x_val = x_val.reshape(-1, 1, 128, 500)

# CREAZIONE ETICHETTE DA [1. , 0.] A [0.0] O [1.0]

label_test,label_val = list(),list() # il resampler trasforma già le etichette in singolo valore per il train set
#for _,col in y_train:
    #label_train.append(col)
#y_train = np.array(label_train)
for _,col in y_val:
    label_val.append(col)
y_val = np.array(label_val)
for _,col in y_test:
    label_test.append(col)
y_test = np.array(label_test)

# Convertiamo i dati numpy in tensori PyTorch
x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
x_val, y_val = torch.from_numpy(x_val), torch.from_numpy(y_val)

# Creiamo i TensorDatasets
train_dataset = TensorDataset(x_train, y_train.reshape(len(y_train),1))
test_dataset = TensorDataset(x_test, y_test.reshape(len(y_test),1))
val_dataset = TensorDataset(x_val, y_val.reshape(len(y_val),1))

# Salvare i dati
torch.save(train_dataset, 'train_dataset.pth')
torch.save(test_dataset, 'test_dataset.pth')
torch.save(val_dataset, 'val_dataset.pth')

# Caricare i dati
train_dataset = torch.load('train_dataset.pth')
test_dataset = torch.load('test_dataset.pth')
val_dataset = torch.load('val_dataset.pth')

# Creare i DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Ora puoi utilizzare train_loader per iterare sul tuo training set durante l'addestramento del modello, e valid_loader per valutare le prestazioni del modello sul validation set.

import torch.nn as nn
import torch.optim as optim
M = 1 #Moltiplicatore
model = Sequential(
    Conv2d(in_channels=1, out_channels=8*M, kernel_size=(3,3), padding=1),
    BatchNorm2d(8*M),  # Aggiungi la batch normalization qui
    ReLU(),
    Dropout(0.4),
    MaxPool2d(kernel_size=2, stride=2),
    Conv2d(in_channels=8*M, out_channels=16*M, kernel_size=(3,3), padding=1),
    BatchNorm2d(16*M),  # Aggiungi la batch normalization qui
    ReLU(),
    Dropout(0.4),
    MaxPool2d(kernel_size=2, stride=2),
    Flatten(),
    Linear(in_features=32*125*16*M, out_features=128*M),
    BatchNorm1d(128*M),  # Aggiungi la batch normalization qui
    ReLU(),
    Dropout(0.4),
    Linear(in_features=128*M, out_features=1),
    Sigmoid()
)

model = model.to(device)

criterion = nn.BCELoss()

lr=0.0000001
optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0.05)

# Initialize lists to save the losses and accuracies
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

num_epochs = 100# Set your number of epochs

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_train_loss = 0
    epoch_train_acc = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.float())

        # Binary classification accuracy
        predicted = outputs > 0.5
        accuracy = (predicted == labels).float().mean()

        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        epoch_train_acc += accuracy.item()

    # Calculate average loss and accuracy
    epoch_train_loss = epoch_train_loss / len(train_loader)
    epoch_train_acc = epoch_train_acc / len(train_loader)

    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_train_loss}, Training Accuracy: {epoch_train_acc}")

    # Validation after each epoch
    model.eval()  # Set the model to evaluation mode
    epoch_valid_loss = 0
    epoch_valid_acc = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())

            # Binary classification accuracy
            predicted = outputs > 0.5
            correct = (predicted == labels).float().sum().item()

            epoch_valid_loss += loss.item() * inputs.size(0)
            epoch_valid_acc += correct

        # Calculate average loss and accuracy
        epoch_valid_loss = epoch_valid_loss / len(valid_loader.dataset)
        epoch_valid_acc = epoch_valid_acc / len(valid_loader.dataset)

        valid_losses.append(epoch_valid_loss)
        valid_accuracies.append(epoch_valid_acc)

        print(f"Validation Loss: {epoch_valid_loss:.4f}, Validation Acc: {epoch_valid_acc:.4f}")

plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.legend()
plt.suptitle("CNN C1 Loss")
plt.title(f"lr={lr}")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlim(0)
plt.ylim(0, 3)
#plt.savefig('CNN_D1/CNN_D1_loss')
plt.show()

plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.legend()
plt.suptitle("CNN D1 Accuracy")
plt.title(f"lr={lr}")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlim(0)
plt.ylim(0,1)
#plt.savefig('CNN_D1/CNN_D1_accuracy')
plt.show()

#torch.save(model.state_dict(), "CNN_D1/CNN_D1.pth")

model.eval()  # Set the model to evaluation mode
test_loss = 0.0
test_correct = 0
all_uncertainties = []
list_threshold = [0,5,0.6,0.7]
balanced_accuracies=[]
percentage_over_threshold = []
over_threshold_test_set = []
num_samples = []
incertezza = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels.float())

        # Binary classification accuracy
        predicted = outputs > 0.5
        correct = (predicted == labels).float().sum().item()
        for logit in outputs.data.cpu().numpy():
            incertezza.append(1 - logit) if logit > 0.5 else incertezza.append(logit)
        test_loss += loss.item() * inputs.size(0)
        test_correct += correct
        incertezza = np.array([incertezza])
        for threshold in list_threshold:
            high_certainty_indices = torch.LongTensor(np.where(np.concatenate(incertezza).ravel() < threshold)[0])
            total_certain_prediction = len(high_certainty_indices)
            all_uncertainties.extend(incertezza[high_certainty_indices])
            high_certainty_predicted = predicted[high_certainty_indices]
            high_certainty_labels = labels[high_certainty_indices]

            correct_certain_samples = (high_certainty_predicted == high_certainty_labels).float().sum().item()

            balanced_accuracy, confusion_matrix = functions.balanced_accuracy_per_threshold(high_certainty_predicted,
                                                                                            high_certainty_labels)

            balanced_accuracies.append(np.round(balanced_accuracy, 2))
            percentage_over_threshold.append(100 * total_certain_prediction / len(test_loader.dataset))

            accuracy = correct_certain_samples / total_certain_prediction

            over_threshold_test_set.append(100 * total_certain_prediction / len(test_loader.dataset))
            num_samples.append(total_certain_prediction)

            samples_percentage = correct_certain_samples / len(test_loader.dataset)

            label = f'Campioni usati: {(100 * total_certain_prediction / len(test_loader.dataset)):.0f}%, ({total_certain_prediction})'
            label_accuracy = f'Balanced Accuracy: {100 * balanced_accuracy:.0f}%'
            fig, ax = plt.subplots()
            ax.boxplot(all_uncertainties)
            ax.text(0.56, 0.06, s=label, transform=ax.transAxes, )
            ax.text(0.56, 0.01, s=label_accuracy, transform=ax.transAxes, )
            plt.title(f'Certainty Threshold = {100 * (1 - threshold):.0f}%')
            plt.suptitle('Predictive Certainty')
            plt.ylabel('Certainty')
            plt.ylim(min(incertezza), max(incertezza))
            plt.show()

    # Calculate average loss and accuracy
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
