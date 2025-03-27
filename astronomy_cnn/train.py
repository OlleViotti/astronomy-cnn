import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import get_data, get_device, EarlyStopping
from model import SpectraCNN

label_names = ["t_eff", "log_g", "fe_h"]  # ["mass", "age", "l_bol", "dist", "t_eff", "log_g", "fe_h", "SNR"]
spectra, labels, spectra_norm, labels_norm, label_names = get_data(label_names = label_names)
num_samples = spectra.shape[0]
input_length = spectra.shape[1]
num_labels = labels.shape[1]

# Split data
train_size = int(0.7 * num_samples)
val_size = int(0.15 * num_samples)
test_size = num_samples - train_size - val_size

# Use TensorDataset to create a dataset
train_dataset, val_dataset, test_dataset = random_split(
    TensorDataset(spectra_norm, labels_norm),
    [train_size, val_size, test_size],
)

# Create DataLoaders
batch_size = 512
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

# Initialize the model
model = SpectraCNN(num_labels=num_labels).to(get_device())
print('Model architecture')
print(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3) # Default weight decay 1e-3. Try 1e-2 and 1e-4
writer = SummaryWriter()

# Learning rate scheduler: reduce LR by factor 0.1 if validation loss doesn't improve for 2 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25)

# Early stopping instance: stop training if no improvement in validation loss for 5 epochs
early_stopping = EarlyStopping(patience=100)

# Train the model
num_epochs = 2000
train_losses, val_losses = [], []

for epoch in range(num_epochs):  # loop through every epoch
    # Training
    model.train()  # The model should be in training mode to use batch normalization and dropout
    train_loss = 0
    for batch_x, batch_y in train_loader:  # loop through every batch
        optimizer.zero_grad()  # set the gradients to zero
        predictions = model(batch_x)  # make a prediction with the current model
        loss = criterion(
            predictions, batch_y
        )  # calculate the loss based on the prediction
        loss.backward()  # calculated the gradients for the given loss
        optimizer.step()  # updates the weights and biases for the given gradients
        train_loss += loss.item()  # calulate loss per batch
    train_loss /= len(train_loader)  # calulate loss per epoch
    train_losses.append(train_loss)

    writer.add_scalar("train_loss", train_loss, epoch)

    # Validation
    model.eval()  # The model should be in eval mode to not use batch normalization and dropout
    val_loss = 0
    with torch.no_grad():  # make sure the gradients are not changed in this step
        for batch_x, batch_y in val_loader:
            predictions = model(batch_x)  # make a prediction with the current model
            loss = criterion(
                predictions, batch_y
            )  # calculate the loss based on the prediction
            val_loss += loss.item()  # calulate loss per batch
    val_loss /= len(val_loader)  # calulate loss per epoch
    val_losses.append(val_loss)
    if val_loss <= min(val_losses):
        torch.save(model.state_dict(), "./models/model_weights.pth")

    writer.add_scalar("val_loss", val_loss, epoch)

    # Update learning rate based on validation loss
    scheduler.step(val_loss)
    
    # Check early stopping condition
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered. Training terminated.")
        break    

    # Print progress
    if epoch % 5 == 0:
        print(
            f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Learning rate = {optimizer.param_groups[0]['lr']}, Lowest val_loss = {early_stopping.best_loss:.4f}, Early stopping counter = {early_stopping.counter}"
        )

def plot_training(train_losses, val_losses, fig_name='training'):
    # Visualize the training progress
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Training loss")
    ax.plot(val_losses, label="Validation loss")
    ax.set_ylabel("MSE")
    ax.set_xlabel("Epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"./plots/{fig_name}.pdf", dpi=600)

plot_training(train_losses, val_losses)

# Load best model
model.load_state_dict(torch.load("./models/model_weights.pth"))

# Test the model
model.eval()  # The model should be in eval mode to not use batch normalization and dropout
test_loss = 0
with torch.no_grad():  # make sure the gradients are not changed in this step
    for batch_x, batch_y in test_loader:
        predictions = model(batch_x)  # make a prediction with the current model
        loss = criterion(
            predictions, batch_y
        )  # calculate the loss based on the prediction
        test_loss += loss.item()  # calulate loss per batch
test_loss /= len(test_loader)  # calculate total loss
print(f"Final Test Loss: {test_loss:.4f}")

# Plot predictions vs target in the scatter plots
def plot_results(model, test_loader, label_names):
    num_labels=len(label_names)
    ncols = int(np.ceil(np.sqrt(num_labels)))
    nrows = int(np.ceil(num_labels / ncols))
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,10))
    model.eval()
    with torch.no_grad():  # make sure the gradients are not changed in this step
        for batch_x, batch_y in test_loader:
            predictions = model(batch_x)  # make a prediction with the current model
            x=predictions.to('cpu')
            y=batch_y.to('cpu')
            for i, ax in enumerate(axs.flatten()):
                if i < num_labels:
                    ax.scatter(x=x[:,i], y=y[:, i], s=2, alpha=0.5, c='blue')

    titles = [f'{l} [normalized]' for l in label_names]
    for i, ax in enumerate(axs.flatten()):
        if i < num_labels:
            ax.set_title(titles[i])
            ax.set_xlabel('prediction')
            ax.set_ylabel('target')
        else:
            ax.set_visible(False)
    fig.tight_layout()
    fig.savefig('./plots/results.pdf', dpi=600)

plot_results(model, test_loader, label_names)

