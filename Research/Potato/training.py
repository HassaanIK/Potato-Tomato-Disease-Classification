from trainer import fit
from model import model
import torch
from data_loader import train_loader, val_loader

lr = 0.001
num_epochs = 20
opt_func = torch.optim.Adam

# Commented out IPython magic to ensure Python compatibility.
# history = fit(num_epochs, lr, model, train_loader, val_loader, opt_func)

# torch.save(model.state_dict(), 'potato_model_statedict__f.pth')