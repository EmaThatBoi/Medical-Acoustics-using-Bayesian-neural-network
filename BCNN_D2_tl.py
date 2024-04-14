import numpy as np
import os
import torch
from torch.nn import Sequential, MaxPool2d, Flatten, ReLU, Sigmoid, BCELoss, Conv2d, Linear, Dropout
from bayesian_torch.models.dnn_to_bnn import get_kl_loss, dnn_to_bnn
from bayesian_torch.utils.util import predictive_entropy, mutual_information
import functions
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X,Y = np.load('X.npy'),np.load('Y.npy')

index = np.random.permutation(len(X))
X = X[index]
Y = Y[index]

x,y= torch.from_numpy(X).float(), torch.from_numpy(Y).float()

from torch.utils.data import TensorDataset, DataLoader

# Assuming x is your input data and y are the labels
tensor_dataset = TensorDataset(x.unsqueeze(1), y)
dataset_length = len(tensor_dataset)
train_length = int(dataset_length * 0.7)
valid_length = (dataset_length - train_length) // 2
test_length = dataset_length - train_length - valid_length

# Randomly split the dataset
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    tensor_dataset,
    lengths=[train_length, valid_length, test_length]
)
batch_size = 128
# Create data loaders for each split
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=97, shuffle=True)

BCNN_model = functions.create_CNN_model()

BCNN_model.load_state_dict(torch.load('CNN_D2/CNN_D2.pth'))

BCNN_model.eval()

const_bnn_prior_parameters = {
    "prior_mu": 0.0,
    "prior_sigma": 1.0,
    "posterior_mu_init": 0.0,
    "posterior_rho_init": -3.0,
    "type": "Flipout",  # Flipout or Reparameterization
    "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
    "moped_delta": 0.2,
}


dnn_to_bnn(BCNN_model, const_bnn_prior_parameters)
# entra qui nei parametri e rimetti a 1 la std
BCNN_model.to(device)
criterion = BCELoss()
lr=0.000025
optimizer = torch.optim.Adam(BCNN_model.parameters(), lr=lr)
num_epochs = 10
num_monte_carlo = 10


train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []
for epoch in range(num_epochs):

    # print weights values distributions during training
    if epoch == 0:
        functions.plot_weights(epoch, BCNN_model)
    if epoch == num_epochs/2:
        functions.plot_weights(epoch, BCNN_model)
    if epoch == num_epochs-1:
        functions.plot_weights(epoch, BCNN_model)

    BCNN_model.train()  # Set the model to training mode
    train_loss = 0.0
    train_correct = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = BCNN_model(inputs)
        kl = get_kl_loss(BCNN_model)
        ce_loss = criterion(outputs, labels)
        loss = ce_loss + kl / len(train_loader.dataset)

        # Binary classification accuracy
        predicted = outputs > 0.5
        correct = (predicted == labels).float().sum().item()

        train_loss += loss.item() * inputs.size(0)
        train_correct += correct

        loss.backward()
        optimizer.step()

    # Calculate average loss and accuracy
    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")

    BCNN_model.eval()
    with torch.no_grad():
        valid_loss = 0.0
        valid_correct = 0
        valid_uncertainties = []
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output_mc = []
            for mc_run in range(num_monte_carlo):
                logits = BCNN_model(inputs)
                # probs = torch.sigmoid(logits)  # ultimo layer nel modello applica già la sigmoide
                output_mc.append(logits)

            output = torch.stack(output_mc)
            pred_mean = output.mean(dim=0)
            predicted = pred_mean > 0.5
            correct = (predicted == labels).float().sum().item()

            kl = get_kl_loss(BCNN_model)
            ce_loss = criterion(pred_mean, labels)
            loss = ce_loss + kl / len(valid_loader.dataset)

            valid_loss += loss.item() * inputs.size(0)
            valid_correct += correct


            uncertainty = predictive_entropy(output.data.cpu().numpy())
            model_uncertainty = mutual_information(output.data.cpu().numpy())

            valid_uncertainties.append((uncertainty, model_uncertainty))

        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_acc = valid_correct / len(valid_loader.dataset)

        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)

        valid_uncertainties = np.array(valid_uncertainties).mean(axis=0)  # calculate mean uncertainties

        print(f"Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.4f}")
        #print('Validation Predictive Uncertainty: ', np.round(valid_uncertainties[0], 4), '\n',
              #'Validation Model Uncertainty: ', np.round(valid_uncertainties[1], 4))

plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.legend()
plt.suptitle("BCNN D2 Loss with transfer learning")
plt.title(f"lr={lr}")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlim(0)
plt.ylim(0, 3)
#plt.savefig('BCNN_D2_tl/BCNN_D2_tl_loss')
plt.show()

plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.legend()
plt.suptitle("BCNN D2 Accuracy with transfer learning")
plt.title(f"lr={lr}")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlim(0)
plt.ylim(0,1)
#plt.savefig('BCNN_D2_tl/BCNN_D2_tl_accuracy')
plt.show()

########################################################

BCNN_model.eval()
test_loss = 0.0
test_correct = 0
correct_certain_samples = 0
total_high_certainty_prediction = 0
all_uncertainties = []
barplot_data = []
total_threshold = np.array([0.35,0.3,0.22,0.17,0.15])  # certainty = 1 - threshold
epistemica_threshold = np.array([0.001,0.0005,0.0002])
aleatoric_threshold = np.array([0.35,0.3,0.22,0.17,0.15])
accuracy = []
over_threshold_test_set = []
balanced_accuracies = []
percentage_over_threshold = []
num_samples = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        output_mc = []
        for mc_run in range(num_monte_carlo):
            logits = BCNN_model(inputs)
            output_mc.append(logits)

        output = torch.stack(output_mc)
        pred_mean = output.mean(dim=0)
        predicted = pred_mean > 0.5  # boolean mask
        correct = (predicted == labels).float().sum().item()

        kl = get_kl_loss(BCNN_model)
        ce_loss = criterion(pred_mean.float(), labels.float())
        loss = ce_loss + kl / test_loader.batch_size

        test_loss += loss.item() * inputs.size(0)
        test_correct += correct

        # uncertainty
        predictive_uncertainty = predictive_entropy(output.data.cpu().numpy())  # incertezza dati (epi+ale)
        model_uncertainty = mutual_information(output.data.cpu().numpy())  # incertezza epistemica
        aleatoric_uncertainty = predictive_uncertainty - model_uncertainty
        #########################
        uncertainty = model_uncertainty  ########Seleziona incertezza
        list_threshold = epistemica_threshold

        for threshold in list_threshold:
            all_uncertainties = []
            high_certainty_indices = np.where(uncertainty < threshold)[0]
            total_certain_prediction = len(high_certainty_indices)
            all_uncertainties.extend(uncertainty[high_certainty_indices])
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

            """label = f'Campioni usati: {(100 * total_certain_prediction / len(test_loader.dataset)):.0f}%, ({total_certain_prediction})'
            label_accuracy = f'Balanced Accuracy: {100 * balanced_accuracy:.0f}%'
            fig, ax = plt.subplots()
            ax.boxplot(all_uncertainties)
            ax.text(0.56, 0.06, s=label, transform=ax.transAxes, )
            ax.text(0.56, 0.01, s=label_accuracy, transform=ax.transAxes, )
            plt.title(f'Certainty Threshold = {100 * (1 - threshold):.0f}%')
            plt.suptitle('Predictive Certainty')
            plt.ylabel('Certainty')
            plt.ylim(min(predictive_uncertainty)-0.05, max(predictive_uncertainty)+0.05)
            plt.show()"""

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.2f}, Test Accuracy: {test_acc:.2f}")

fig, ax = plt.subplots()
ax.plot(list((1 - list_threshold) * 100), (100 * np.array(balanced_accuracies)), 'o-r')

# Add annotations for number of samples
for i, certainty in enumerate(list((1 - list_threshold) * 100)):
    ax.annotate(f'{percentage_over_threshold[i]:.0f}%,{num_samples[i]}', (certainty + 0.003, 100 * balanced_accuracies[i] + 0.002))

# Set the y-axis ticks to be the discrete values you want
ax.set_yticks([100 * value for value in balanced_accuracies])
ax.set_xticks([ value for value in list((1 - list_threshold) * 100)])

plt.xlabel('Certezza %')
plt.ylabel('Balanced Accuracy %')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.title('Certezza epistemica')
plt.show()


