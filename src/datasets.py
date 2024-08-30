"""

@author: Jinal Shah

This script will contain classes
that represent the dataset(s) that 
I will need.

"""
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import pandas as pd
import matplotlib.pyplot as plt

# Class for the State Farm Distracted Driver Detection dataset
class StateFarmDD(Dataset):
    # Constructor
    def __init__(self,annotations_path,path_prefix='../data/imgs/train',transform_pipeline=None):
        self.annotations = pd.read_csv(annotations_path)
        self.transformation_pipeline = transform_pipeline
        self.path_prefix = path_prefix
        self.label_to_int_dict = {'c0':0,'c1':1,'c2':2,'c3':3,'c4':4,'c5':5,'c6':6,'c7':7,'c8':8,'c9':9}
    
    # Method to get the length of the dataset
    def __len__(self):
        return len(self.annotations)
    
    # Method to get the item at a particular index
    def __getitem__(self,index):
        label = self.annotations.iloc[index,1]
        image_name = self.annotations.iloc[index,2]
        image = plt.imread(f'{self.path_prefix}/{label}/{image_name}')

        # Throwing image through pipeline if it exists
        if self.transformation_pipeline:
            transformed_image = self.transformation_pipeline(image.copy()).squeeze(0)
            return transformed_image, self.label_to_int_dict[label]
        else:
            return image, self.label_to_int_dict[label]

# Testing the dataset class(es)
if __name__ == '__main__':
    # Testing the StateFarmDD class
    transformation_pipeline = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(),
        v2.CenterCrop(500),
    ])
    statefarm_dd = StateFarmDD('../data/training.csv',transformation_pipeline)
    image, label = statefarm_dd[0]
    print(image.shape)
    print(label)
    plt.imshow(image,cmap='gray')
    plt.axis('off')
    plt.show()