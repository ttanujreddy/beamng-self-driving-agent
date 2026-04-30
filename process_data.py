"""
Author(s): Matt Gallenberger
Class: CS450-01
Date: 04/29/26
"""

import pandas as pd
from PIL import Image
import numpy  as np
import os

# Creates a folder for the processed frames.
os.makedirs("processed_frames", exist_ok=True)

# Imports the raw csv.
df = pd.read_csv('raw_data.csv')

# Deletes any row that contain null values.
df.dropna(inplace=True)

# Loops through the "Frame" in the df column and normalizes each frame data.
for filename in df['Frame']:
   image = Image.open(filename) # load original
   normalized = np.array(image) / 255 # normalize to 0-1
   new_filename = filename.replace("frames/", "processed_frames/") # build new file path
   Image.fromarray((normalized * 255).astype('uint8')).save(new_filename) # save

df['Frame'] = df['Frame'].str.replace('frames/', 'processed_frames/')

# Converts the values from the raw csv to amounts between 1 and 0 for the neural network.
df['Speed'] = (df['Speed'] - df['Speed'].min()) / (df['Speed'].max() - df['Speed'].min())
df['Steering'] = (df['Steering'] - df['Steering'].min()) / (df['Steering'].max() - df['Steering'].min())
df['Throttle'] = (df['Throttle'] - df['Throttle'].min()) / (df['Throttle'].max() - df['Throttle'].min())
df['Brake'] = (df['Brake'] - df['Brake'].min()) / (df['Brake'].max() - df['Brake'].min())

# Saves clean data to new csv.
df.to_csv('processed_data.csv', index=False)

