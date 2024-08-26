"""

@author: Jinal Shah

This script holds a class 
to log the metrics of each 
model in a csv file.

"""
import csv
import os
from sklearn.metrics import log_loss, accuracy_score
import numpy as np

# Class to log the metrics of each model into a csv file
class MLMetricsLogger:
    def __init__(self,filename='../models/metrics.csv'):
        self.filename = filename
        self.columns = ['model_id','model_name','train_CE_loss','validation_CE_loss','train_acc','validation_acc']
        
        # Creating the file if it doesn't exist
        if not os.path.exists(filename):
            with open(self.filename,'w',newline='') as f:
                writer = csv.DictWriter(f,fieldnames=self.columns)
                writer.writeheader()
    
    # Function to calculate the metrics
    def calculate_metrics(self,model_id,model_name,training_targets,training_predictions,val_targets,val_predictions):
        
        # Getting the metrics
        train_CE_loss = log_loss(training_targets,training_predictions)
        validation_CE_loss = log_loss(val_targets,val_predictions)

        # Getting the accuracy
        train_class_preds = np.array(training_predictions).argmax(axis=1)
        valid_class_preds = np.array(val_predictions).argmax(axis=1)
        train_acc = accuracy_score(training_targets,train_class_preds)
        valid_acc = accuracy_score(val_targets,valid_class_preds)
        
        return {'model_id':model_id,'model_name':model_name,'train_CE_loss':train_CE_loss,
                'train_acc':train_acc,'validation_CE_loss':validation_CE_loss,'validation_acc':valid_acc}
    
    # Function to log metrics
    def log_metrics(self,metrics):
        metrics_table = self.read_metrics()
        model_ids = [int(row['model_id']) for row in metrics_table]
        if int(metrics['model_id']) in model_ids:
            raise ValueError('Model ID already exists in the metrics table. Please use a different model ID.')
        else:
            with open(self.filename,'a',newline='') as f:
                writer = csv.DictWriter(f,fieldnames=self.columns)
                writer.writerow(metrics)
    
    # Function to read metrics -> returns a list 
    def read_metrics(self):
        metrics = []

        with open(self.filename,'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics.append(row)
        
        return metrics