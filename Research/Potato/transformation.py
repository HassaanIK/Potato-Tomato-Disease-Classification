from torchvision import transforms as tt

# Define the transformations to be applied to the images
transforms = tt.Compose([
    tt.Resize((224, 224)),
    tt.RandomHorizontalFlip(p=0.5),
    tt.RandomVerticalFlip(p=0.5),
    tt.RandomRotation(20),
    tt.ToTensor()
])