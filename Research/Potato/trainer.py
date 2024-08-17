from torch.optim.lr_scheduler import ReduceLROnPlateau
from earlystopping import EarlyStopping
from evaluater import evaluate
import torch


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

        # LR Scheduler step
        scheduler.step(result['val_loss'])
        early_stopping(result['val_loss'], model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return history