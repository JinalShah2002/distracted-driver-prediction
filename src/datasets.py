"""

@author: Jinal Shah

This script will contain classes
that represent the dataset(s) that 
I will need.

"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2

# Class for the State Farm Distracted Driver Detection dataset
class StateFarmDD(Dataset):
    # Constructor
    def __init__(self,annotations_path,transform_pipeline=None):
        self.annotations = pd.read_csv(annotations_path)
        self.transformation_pipeline = transform_pipeline
    
    # Method to get the length of the dataset
    def __len__(self):
        return len(self.annotations)
    
    # Method to get the item at a particular index
    def __getitem__(self,index):
        label = self.annotations.iloc[index,1]
        image_name = self.annotations.iloc[index,2]
        image = cv2.imread(f'../data/imgs/train/{label}/{image_name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Throwing image through pipeline if it exists
        if self.transformation_pipeline:
            image = self.transformation_pipeline(image)
        
        return image, label