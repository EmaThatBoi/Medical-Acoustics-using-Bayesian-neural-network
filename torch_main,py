import numpy as np
import os
import torch
from torch.nn import Sequential, MaxPool2d, Flatten, ReLU, Sigmoid, BCELoss
from torchbnn import BayesConv2d, BayesLinear, BayesBatchNorm2d, BKLLoss


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

#per vedere la distribuzione della lunghezza degli spettrogrammi
leng = [len(x_data[i][0]) for i in range(len(x_data))] #lunghezza spettrogrammi
plt.hist(leng, bins='auto')  # 'auto' automatically determines the number of bins
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.xlim(0,100)
plt.title('Distribution of Matrix Lengths')
plt.show()

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
"""

#x_data,y_data = np.load('dataset.npy'),np.load('labels.npy')
#x_data = x_data.reshape(x_data.shape[0], -1)  # shape diventa (920, 128*500)
#ratio = {0: 322, 1: 322}
#rus = RandomUnderSampler(sampling_strategy=ratio, random_state=0)
#x_data, y_data = rus.fit_resample(x_data, y_data)
#x_data = x_data.reshape(x_data.shape[0], 128, 500)

#np.save(file='X',arr=x_data)
#np.save(file='Y',arr=y_data)

X,Y = np.load('X.npy'),np.load('Y.npy')

index = np.random.permutation(len(X))
X = X[index]
Y = Y[index]

x,y= torch.from_numpy(X).float(), torch.from_numpy(Y).float()

from torch.utils.data import TensorDataset, DataLoader

# Assuming x is your input data and y are the labels
tensor_dataset = TensorDataset(x.unsqueeze(1), y)

batch_size = 32  # Set your batch size

dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

model = Sequential(
    BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    BayesConv2d(prior_mu=0, prior_sigma=0.1, in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    Flatten(),
    BayesLinear(prior_mu=0, prior_sigma=0.1,in_features=32*125*32, out_features=128 ),
    ReLU(),
    BayesLinear(prior_mu=0, prior_sigma=0.1,in_features=128, out_features=1),
    Sigmoid()
)

criterion = BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x = x.unsqueeze(1)
num_epochs = 10
bce_loss = BCELoss()
kl_loss = BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(dataloader):
        #inputs = inputs.to(device)  # If using GPU
        #labels = labels.to(device)  # If using GPU

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        bce = bce_loss(outputs, labels)
        kl = kl_loss(model)
        loss = bce + kl_weight*kl

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        predicted = torch.round(outputs.data)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch+1, num_epochs, loss.item(), (accuracy * 100)))
