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
train_dataset = torch.load('train_dataset.pth')
test_dataset = torch.load('test_dataset.pth')
val_dataset = torch.load('val_dataset.pth')

# Creare i DataLoader
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=210, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=140, shuffle=True)

BCNN_model = functions.create_CNN_model()

BCNN_model.load_state_dict(torch.load('CNN_D1/CNN_D1.pth'))

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
BCNN_model.to(device)
criterion = BCELoss()
lr=0.00025
optimizer = torch.optim.Adam(BCNN_model.parameters(), lr=lr)
num_epochs = 100
num_monte_carlo = 50


train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []
for epoch in range(num_epochs):

    BCNN_model.train()  # Set the model to training mode
    train_loss = 0.0
    train_correct = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = BCNN_model(inputs)
        kl = get_kl_loss(BCNN_model)
        ce_loss = criterion(outputs, labels.float())
        loss = ce_loss + kl / batch_size

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
            ce_loss = criterion(pred_mean, labels.float())
            loss = ce_loss + kl / batch_size

            valid_loss += loss.item() * inputs.size(0)
            valid_correct += correct

            predictive_uncertainty = predictive_entropy(output.data.cpu().numpy())
            model_uncertainty = mutual_information(output.data.cpu().numpy())

            valid_uncertainties.append((predictive_uncertainty, model_uncertainty))

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
plt.suptitle("BCNN D1 transfer learning Loss")
plt.title(f"lr={lr}")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlim(0,100)
plt.ylim(0, 3)
plt.savefig('BCNN_D1_tl/BCNN_D1_tl_loss')
plt.show()

plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.legend()
plt.suptitle("BCNN D1 transfer learning Accuracy")
plt.title(f"lr={lr}")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xlim(0,100)
plt.ylim(0,1)
plt.savefig('BCNN_D1_tl/BCNN_D1_tl_accuracy')
plt.show()

BCNN_model.eval()
test_loss = 0.0
test_correct = 0
correct_certain_samples = 0
total_high_certainty_prediction = 0
all_uncertainties = []
barplot_data = []
total_threshold = np.array([0.38,0.35,0.32, 0.3, 0.25])  # certainty = 1 - threshold
epistemic_threshold = np.array([0.006,0.0035,0.0017,0.00075,0.00025])
aleatoric_threshold = np.array([0.38,0.35,0.32, 0.3, 0.25])
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
        list_threshold = epistemic_threshold

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
    ax.annotate(f'{percentage_over_threshold[i]:.0f}%,{num_samples[i]}', (certainty + 0.02, 100 * balanced_accuracies[i] + 0.02))

# Set the y-axis ticks to be the discrete values you want
ax.set_yticks([100 * value for value in balanced_accuracies])
ax.set_xticks([ value for value in list((1 - list_threshold) * 100)])

plt.xlabel('Certezza %')
plt.ylabel('Balanced Accuracy %')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.title('Certezza epistemica')
plt.show()

