"""

@author: Jinal Shah

A script to build and train 
the baseline model

Baseline Model: a random classifier

"""
import pandas as pd
import torch
import torch.nn.functional as F
from metricsLogger import MLMetricsLogger

# Getting the data
train_data = pd.read_csv('../data/training.csv')
valid_data = pd.read_csv('../data/validation.csv')

# Mapping labels to integers
mapping = {'c0':0,'c1':1,'c2':2,'c3':3,'c4':4,'c5':5,'c6':6,'c7':7,'c8':8,'c9':9}

train_data['target'] = train_data['classname'].map(mapping)
valid_data['target'] = valid_data['classname'].map(mapping)

# Generating the predictions -> random numbers from a normal distribution
train_pred = torch.randn((train_data.shape[0],10))
valid_pred = torch.randn((valid_data.shape[0],10))

# Normalizing the predictions
train_pred = F.softmax(train_pred,dim=1).numpy()
valid_pred = F.softmax(valid_pred,dim=1).numpy()

# Getting the metrics
logger = MLMetricsLogger()
metrics = logger.calculate_metrics('1','Baseline Model (Random Classifer)',train_data['target'],train_pred,valid_data['target'],valid_pred)
print(metrics)

# Logging the metrics
logger.log_metrics(metrics)