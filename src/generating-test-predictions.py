"""

@author: Jinal Shah

This script generates the test predictions
for the test set.

"""
import pandas as pd
from datasets import StateFarmDD
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

# Creating the testing dataset
sample_submission = pd.read_csv('../data/sample_submission.csv')
test_df = pd.DataFrame({
    "subject": ['None'] * len(sample_submission),
    "classname": ['c10'] * len(sample_submission),
    "img": sample_submission['img']
})

# Saving the test_df to a csv file
test_df.to_csv('../data/test_df.csv', index=False)

# Creating the testing dataset
path_prefix = '../data/imgs/test'
transformation_pipeline = v2.Compose([
        v2.ToImage(),
        v2.Resize([128,128]),
        v2.ToDtype(torch.float32, scale=True),
])
test_dataset = StateFarmDD('../data/test_df.csv', path_prefix=path_prefix, transform_pipeline=transformation_pipeline, test=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define the model architecture
print('Loading the model...')
model = nn.Sequential(
    nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.BatchNorm2d(64),
    nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.BatchNorm2d(128),
    nn.Dropout(p=0.5),
    nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(256),
    nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.BatchNorm2d(512),
    nn.Dropout(p=0.5),
    nn.Flatten(),
    nn.Linear(131072,500),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(500,10)
)

# Load the model weights
model.load_state_dict(torch.load('../models/cnn_3.pth'))
model.eval()
print('Model loaded successfully!')
print()

# Generate predictions
predictions = []
with torch.no_grad():
    for images, _ in tqdm(test_loader, desc='Generating predictions'):
        outputs = model(images)

        # Getting output probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)

        # Adding probabilities to the predictions list
        predictions.extend(probs.numpy())

# Creating the test predictions dataframe
test_predictions = pd.DataFrame({
    "img": sample_submission['img'],
    "c0": [pred[0] for pred in predictions],
    "c1": [pred[1] for pred in predictions],
    "c2": [pred[2] for pred in predictions],
    "c3": [pred[3] for pred in predictions],
    "c4": [pred[4] for pred in predictions],
    "c5": [pred[5] for pred in predictions],
    "c6": [pred[6] for pred in predictions],
    "c7": [pred[7] for pred in predictions],
    "c8": [pred[8] for pred in predictions],
    "c9": [pred[9] for pred in predictions],
})

# Saving the test predictions dataframe
test_predictions.to_csv('../data/test_predictions.csv', index=False)