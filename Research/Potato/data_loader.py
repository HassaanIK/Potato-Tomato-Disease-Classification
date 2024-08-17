from torch.utils.data import DataLoader
from data_splitter import train_set, val_set

batch_size = 32
train_loader = DataLoader(train_set, batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size)