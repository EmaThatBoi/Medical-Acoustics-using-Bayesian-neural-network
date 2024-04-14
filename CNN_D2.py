import numpy as np
import os
import torch
import functions
from torch.utils.data import TensorDataset, DataLoader
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torch.nn import Sequential, MaxPool2d, Flatten, ReLU, Sigmoid, BCELoss, Conv2d, Linear, Dropout
from bayesian_torch.models.dnn_to_bnn import get_kl_loss, dnn_to_bnn
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# script pre-processing.py -> spettri suddivisi per diagnosi nelle corrispondenti directory
#                             ndarray paddati con 0.

# Carica dati
x_data,y_data = np.load('unofficial set/dataset.npy'),np.load('unofficial set/labels.npy')

#per vedere la distribuzione della lunghezza degli spettrogrammi
"""leng = [len(x_data[i][0]) for i in range(len(x_data))] #lunghezza spettrogrammi
plt.hist(leng, bins='auto')  # 'auto' automatically determines the number of bins
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.xlim(0,100)
plt.title('Distribution of Matrix Lengths')
plt.show()"""

"""
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
y_test = y_data[num_train_samples+num_val_samples:]""" # multi classe

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
x_train, x_val, y_train,y_val = train_test_split(x_train,y_train, test_size=0.2, random_state=42)

#BILANCIARE SOLO TRAIN SET,
x_train = x_train.reshape(x_train.shape[0], -1)  # shape diventa (920, 128*500)
ratio = {0: 208, 1: 208}
rus = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
x_train, y_train = rus.fit_resample(x_train, y_train)
x_train = x_train.reshape(x_train.shape[0], 128, 500)

x_train = x_train.reshape(-1, 1, 128, 500)
x_test = x_test.reshape(-1, 1, 128, 500)
x_val = x_val.reshape(-1, 1, 128, 500)

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
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


M = 2 #Moltiplicatore
model = Sequential(
    Conv2d(in_channels=1, out_channels=8*M, kernel_size=(3,3), padding=1),
    ReLU(),
    Dropout(0.4),
    MaxPool2d(kernel_size=2, stride=2),
    Conv2d( in_channels=8*M, out_channels=16*M, kernel_size=(3,3), padding=1),
    ReLU(),
    Dropout(0.4),
    MaxPool2d(kernel_size=2, stride=2),
    Flatten(),
    Linear(in_features=32*125*16*M, out_features=128*M ),
    ReLU(),
    Dropout(0.4),
    Linear(in_features=128*M, out_features=1),
    Sigmoid()
)
model = model.to(device)
criterion = BCELoss()
lr=0.000005

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.05)


num_epochs = 100  # Set your number of epochs

train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

num_epochs = 100 # Set your number of epochs

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
plt.suptitle("CNN D2 Loss")
plt.title(f"lr={lr}")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlim(0)
plt.ylim(0, 3)
plt.savefig('CNN_D2_loss')
plt.show()

plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.legend()
plt.suptitle("CNN D2 Accuracy")
plt.title(f"lr={lr}")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlim(0)
plt.ylim(0,1)
plt.savefig('CNN_D2_accuracy')
plt.show()

torch.save(model.state_dict(), "CNN_D2/CNN_D2.pth")

model.eval()  # Set the model to evaluation mode
test_loss = 0.0
test_correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels.float())

        # Binary classification accuracy
        predicted = outputs > 0.5
        correct = (predicted == labels).float().sum().item()

        test_loss += loss.item() * inputs.size(0)
        test_correct += correct

    # Calculate average loss and accuracy
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
