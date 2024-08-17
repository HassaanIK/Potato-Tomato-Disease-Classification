from torchvision.datasets import ImageFolder
from transformation import transforms

data_path = 'Data'
dataset = ImageFolder(data_path, transform=transforms)