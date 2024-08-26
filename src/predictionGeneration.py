"""

@author: Jinal Shah

This file generates the predictions
for the test set.

"""
import os
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

transformation_pipeline = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(),
        v2.Resize([100,100]),
        v2.ToDtype(torch.float32, scale=True),
])

pca_model = joblib.load('../models/pca.pkl')
model = joblib.load('../models/model_2.pkl')
prediction_pipeline = Pipeline([
    ('pca',pca_model),
    ('model',model)
])

# Getting the predictions
predictions = []
image_paths = os.listdir('../data/imgs/test')
n = len(image_paths)
# n = 5
for i in tqdm(range(n)):
    image = plt.imread(f'../data/imgs/test/{image_paths[i]}')
    transformed_image = transformation_pipeline(image)
    transformed_image = torch.flatten(transformed_image,start_dim=1).numpy()
    row = [image_paths[i]]
    row.extend(prediction_pipeline.predict_proba(transformed_image).squeeze(axis=0).tolist())
    predictions.append(row)
predictions = np.array(predictions)
predictions_df = pd.DataFrame(predictions,columns=['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
predictions_df.to_csv('../data/predictions.csv',index=False)