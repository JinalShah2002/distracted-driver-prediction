"""

@author: Jinal Shah

This file will train 
an artificial neural network

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow 
from metricsLogger import MLMetricsLogger
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset

# Getting the data
training_data = pd.read_parquet('../data/train.parquet')
validation_data = pd.read_parquet('../data/valid.parquet')

# Getting the features and labels
X_train, y_train = training_data.drop('label', axis=1), torch.LongTensor(training_data['label'].values)
X_valid, y_valid = validation_data.drop('label', axis=1), torch.LongTensor(validation_data['label'].values)

# Getting the PCA model and running PCA
pca = joblib.load('../models/pca.pkl')
X_train_processed = torch.FloatTensor(pca.transform(X_train))
X_valid_processed = torch.FloatTensor(pca.transform(X_valid))

# Putting the data into a DataLoader
train_data = TensorDataset(X_train_processed, y_train)
valid_data = TensorDataset(X_valid_processed, y_valid)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True)

# Defining the model
model = torch.nn.Sequential(
    torch.nn.Linear(196,100),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(100,50),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(50,10)
)
loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
training_history_loss = []
validation_history_loss = []
train_history_accuracy = []
valid_history_accuracy = []
epochs = 100
current_count = 0
early_stopping_threshold = 1e-4
early_stopping_count = 5
best_val_loss = float('inf')

# Training the model
for epoch in range(epochs):
    model.train()
    train_loss = 0
    valid_loss = 0
    train_accuracy = 0
    valid_accuracy = 0
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Computing the accuracy
        y_pred_labels = torch.argmax(y_pred, dim=1)
        train_accuracy += torch.sum(y_pred_labels == y_batch).item()

    model.eval()
    for X_batch, y_batch in valid_loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        valid_loss += loss.item()

        # Computing the accuracy
        y_pred_labels = torch.argmax(y_pred, dim=1)
        valid_accuracy += torch.sum(y_pred_labels == y_batch).item()
    
    train_loss /= len(train_loader.dataset)
    valid_loss /= len(valid_loader.dataset)
    train_accuracy /= len(train_loader.dataset)
    valid_accuracy /= len(valid_loader.dataset)
    training_history_loss.append(train_loss)
    validation_history_loss.append(valid_loss)
    train_history_accuracy.append(train_accuracy)
    valid_history_accuracy.append(valid_accuracy)

    # Early stopping
    if epoch > 0:
        if validation_history_loss[-1] - best_val_loss > early_stopping_threshold:
            current_count += 1
        else:
            best_val_loss = validation_history_loss[-1]
            current_count = 0
    else:
        best_val_loss = validation_history_loss[-1]
        
    if current_count == early_stopping_count:
        print('Stopping training due to early stopping!!!')
        break
    elif current_count == 0:
        # Saving the best model
        torch.save(model.state_dict(), '../models/ann.pth')
    
    print('-----------------------------------')
    print(f'Epoch {epoch}')
    print(f'Training Loss: {round(train_loss,4)}')
    print(f'Validation Loss: {round(valid_loss,4)}')
    print(f'Training Accuracy: {round(train_accuracy*100,4)}%')
    print(f'Validation Accuracy: {round(valid_accuracy*100,4)}%')
    print()
    print(f'Best Validation Loss: {round(best_val_loss,4)}')
    print('-----------------------------------')
    print()

# Saving Plots for the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(training_history_loss, label='Training Loss')
plt.plot(validation_history_loss, label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.savefig('../models/ann_loss.png')
plt.close()

# Saving plot for the training and validation accuracy 
plt.figure(figsize=(10, 6))
plt.plot(train_history_accuracy, label='Training Accuracy')
plt.plot(valid_history_accuracy, label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.savefig('../models/ann_accuracy.png')
plt.close()

# Making the predictions for the training & validation for metric logging
model.load_state_dict(torch.load('../models/ann.pth')) # loading the best model
model.eval()
logger = MLMetricsLogger()
train_pred = []
valid_pred = []
train_truth = []
valid_truth = []

# Running through data loaders to store the predictions
for X_batch, y_batch in train_loader:
    y_pred = torch.nn.functional.softmax(model(X_batch),dim=1)
    train_pred.extend(y_pred.detach().numpy())
    train_truth.extend(y_batch.numpy())

for X_batch, y_batch in valid_loader:
    y_pred = torch.nn.functional.softmax(model(X_batch),dim=1)
    valid_pred.extend(y_pred.detach().numpy())
    valid_truth.extend(y_batch.numpy())

# Printing out the metrics 
metrics = logger.calculate_metrics('7',"ANN",train_truth,np.array(train_pred),valid_truth,np.array(valid_pred))
print(metrics)

# Asking if metrics should be logged
log_metrics = input('Do you want to log the metrics? (y/n): ')
if log_metrics.lower() == 'y':
    logger.log_metrics(metrics)
    print('Metrics have been logged!!!')
else:
    print('Metrics have not been logged!!!')