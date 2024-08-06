"""

@author: Jinal Shah

A script to build and train 
the baseline model

"""
from datasets import StateFarmDD
from metricsLogger import MLMetricsLogger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Getting the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defining annotation paths
training_annotations = '../data/training.csv'
validation_annotations = '../data/validation.csv'

# Tranformation pipeline
transformation_pipeline = v2.Compose([v2.ToImage()])

# Getting the datasets
print('Loading the datasets...')
training_dataset = StateFarmDD(training_annotations,transformation_pipeline)
validation_dataset = StateFarmDD(validation_annotations,transformation_pipeline)
print('Datasets loaded successfully!')
print()

# Loading the datasets into the dataloaders
print('Loading the datasets into the dataloaders...')
training_loader = DataLoader(training_dataset,batch_size=128,shuffle=True)
valid_loader = DataLoader(validation_dataset,batch_size=128,shuffle=True)
print('Datasets loaded into the dataloaders successfully!')
print()

# Defining the model
baseline_model = nn.Sequential(
    nn.Conv2d(3,16,3,stride=3),
    nn.ReLU(),
    nn.MaxPool2d(3,3),
    nn.Conv2d(16,16,3,stride=3),
    nn.ReLU(),
    nn.MaxPool2d(3,3),
    nn.Flatten(),
    nn.Linear(560,10,bias=True)
)

# Training the model
print('Training the model...')
baseline_model.to(device)
optimizer = torch.optim.Adam(baseline_model.parameters(),lr=0.001)
loss_fn = nn.CrossEntropyLoss(reduction='sum')
training_history_loss = []
validation_history_loss = []

for epoch in range(10):
    print(f'Epoch {epoch+1}')
    training_loss = 0.0
    validation_loss = 0.0
    for i, data in enumerate(tqdm(training_loader)):
        inputs, labels = data
        inputs = inputs.float()
        labels = torch.tensor(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = baseline_model(inputs)
        loss = loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    
    for i, data in enumerate(tqdm(valid_loader)):
        inputs, labels = data
        inputs = inputs.float()
        labels = torch.tensor(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = baseline_model(inputs)
        loss = loss_fn(outputs,labels)
        validation_loss += loss.item()
    
    average_training_loss = training_loss/len(training_dataset)
    average_validation_loss = validation_loss/len(validation_dataset)
    training_history_loss.append(average_training_loss)
    validation_history_loss.append(average_validation_loss)
    print(f'Epoch {epoch+1} -> Training Loss: {average_training_loss} Validation Loss: {average_validation_loss}')
    print()

print('Model trained successfully!')
print()

# Saving the final model
print('Saving the model...')
baseline_model.to('cpu') # Putting model back on to CPU to save it
model_id = "1"
model_name = "baseline_model"
torch.save(baseline_model.state_dict(),f'../models/{model_name}_{model_id}.pth')
baseline_model.to(device) # Putting model back on to GPU, if available
print('Model saved successfully!')
print()

# Plotting the learning curves
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(training_history_loss,label='Training Loss')
plt.plot(validation_history_loss,label='Validation Loss')
plt.legend()
plt.savefig(f'../models/{model_id}_{model_name}_loss_curves.png')

# Getting predictions
print('Getting predictions...')
training_predictions = []
validation_predictions = []
training_targets = []
validation_targets = []

# Getting training predictions
for i, data in enumerate(tqdm(training_loader)):
    inputs, labels = data
    inputs = inputs.float()
    labels = torch.tensor(labels)
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = baseline_model(inputs)

    # Putting outputs through softmax
    outputs = nn.functional.softmax(outputs,dim=1).tolist()
    training_predictions.extend(outputs)
    training_targets.extend(labels.tolist())

# Getting validation predictions
for i, data in enumerate(tqdm(valid_loader)):
    inputs, labels = data
    inputs = inputs.float()
    labels = torch.tensor(labels)
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = baseline_model(inputs)
    outputs = nn.functional.softmax(outputs,dim=1).tolist()
    validation_predictions.extend(outputs)
    validation_targets.extend(labels.tolist())
print('Predictions obtained successfully!')
print()

# Logging the metrics
print('Logging the metrics...')
metrics_logger = MLMetricsLogger()
metrics_logger.log_metrics(model_id,model_name,training_targets,training_predictions,validation_targets,validation_predictions)

# Reading the metrics
metrics = metrics_logger.read_metrics()
print(pd.DataFrame(metrics))
print('Metrics logged successfully!')
print()