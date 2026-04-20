import pandas as pd

# Imports the raw csv.
df = pd.read_csv('raw_data.csv')

# Deletes any row that contain null values.
df.dropna(inplace=True)


# Converts the values from the raw csv to amounts between 1 and 0 for the neural network.
df['Speed'] = (df['Speed'] - df['Speed'].min()) / (df['Speed'].max() - df['Speed'].min())
df['Steering'] = (df['Steering'] - df['Steering'].min()) / (df['Steering'].max() - df['Steering'].min())
df['Throttle'] = (df['Throttle'] - df['Throttle'].min()) / (df['Throttle'].max() - df['Throttle'].min())
df['Brake'] = (df['Brake'] - df['Brake'].min()) / (df['Brake'].max() - df['Brake'].min())

# Saves clean data to new csv.
df.to_csv('processed_data.csv', index=False)

