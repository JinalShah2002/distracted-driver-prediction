# Distracted Driver Prediction
Using machine learning to predict whether a driver is distracted or not.

## Problem
According to the CDC motor vehicle safety division, one in five car accidents is caused by a distracted driver. Sadly, these accidents result in 425,000 people injured and 3,000 people killed by distracted driving every year. Clearly, distracted driving is leading to harmful outcomes, and we need to find a way to minimize it. Using machine learning & computer vision, I built a system that can detect determine if a driver is distracted.

## How it works
The system uses a convolutional neural network to detect if a driver is distracted. The model takes in video feed and feeds each frame into the model to get a prediction. 

## How to run
1. Clone the repository
2. Install the dependencies using Poetry:
   - If you don't have Poetry installed, first install it by following the instructions on the Poetry website (https://python-poetry.org/docs/#installation).
   - Navigate to the project directory in your terminal.
   - Run `poetry install` to install all dependencies.
3. Navigate to the `src` directory.
4. Run the `main.py` file with the following command:
   ```
   python main.py
   ```
5. The video feed should open in a window.

Note, the model I am using is not available in this repository since it is too large. However, I have included the model architecture and the training notebook. 