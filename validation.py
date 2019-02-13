import os

import torch
import torchvision
from torchvision import transforms
import torch.utils.data


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = './data'
BATCH_SIZE = 128

MODEL_PATH = './models'
MODEL_NAME = 'alexnet.pth'

# Create model
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

transform = transforms.Compose([
    transforms.RandomCrop(36, padding=4),
    transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# Load data
dataset = torchvision.datasets.CIFAR100(root=WORK_DIR,
                                        download=True,
                                        train=False,
                                        transform=transform)

dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)


def main():
    print(f"Val numbers:{len(dataset)}")

    # Load model
    if device == 'cuda':
        model = torch.load(MODEL_PATH + MODEL_NAME).to(device)
    else:
        model = torch.load(MODEL_PATH + MODEL_NAME, map_location='cpu')
    model.eval()

    correct = 0.
    total = 0
    for images, labels in dataset_loader:
        # to GPU
        images = images.to(device)
        labels = labels.to(device)
        # print prediction
        outputs = model(images)
        # equal prediction and acc
        _, predicted = torch.max(outputs.data, 1)
        # val_loader total
        total += labels.size(0)
        # add correct
        correct += (predicted == labels).sum().item()

    print(f"Acc: {correct / total:.4f}.")


if __name__ == '__main__':
    main()
