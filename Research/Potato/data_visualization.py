import matplotlib.pyplot as plt
import numpy as np
from data_transformer import dataset
from data_loader import train_loader

# Define a function to display images
def show_images(images, labels, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(20, 5))
    for i in range(num_images):
        axes[i].imshow(np.transpose(images[i], (1, 2, 0)))
        axes[i].set_title(labels[i])
        axes[i].axis('off')
    plt.show()

# Get some sample images and labels from the dataset
num_images_to_display = 5
sample_indices = np.random.choice(len(dataset), num_images_to_display, replace=False)
sample_images = [dataset[i][0] for i in sample_indices]
sample_labels = [dataset.classes[dataset[i][1]] for i in sample_indices]

# Display the sample images with their labels
# show_images(sample_images, sample_labels, num_images=num_images_to_display)

from torchvision.utils import make_grid

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=12).permute(1, 2, 0))
        break

# show_batch(train_loader)