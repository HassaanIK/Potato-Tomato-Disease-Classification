from torch.utils.data import random_split
from data_transformer import dataset

# Define the sizes of the train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Split the dataset into train and validation sets
train_set, val_set = random_split(dataset, [train_size, val_size])