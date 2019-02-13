import os

import time

import torch
import torchvision
from torchvision import transforms
import torch.utils.data

# first train run this code
from model import AlexNet
# incremental training comments out that line of code.

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = './data'
NUM_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-4

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
                                        train=True,
                                        transform=transform)

dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)


def main():
    print(f"Train numbers:{len(dataset)}")

    # first train run this line
    model = AlexNet().to(device)
    # Load model
    # if device == 'cuda':
    #     model = torch.load(MODEL_PATH + MODEL_NAME).to(device)
    # else:
    #     model = torch.load(MODEL_PATH + MODEL_NAME, map_location='cpu')
    # cast
    cast = torch.nn.CrossEntropyLoss().to(device)
    # Optimization
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-8)
    step = 1
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        # cal one epoch time
        start = time.time()

        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = cast(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Step [{step * BATCH_SIZE}/{NUM_EPOCHS * len(dataset)}], "
                  f"Loss: {loss.item():.8f}.")
            step += 1

        # cal train one epoch time
        end = time.time()
        print(f"Epoch [{epoch}/{NUM_EPOCHS}], "
              f"time: {end - start} sec!")

        # Save the model checkpoint
        torch.save(model, MODEL_PATH + MODEL_NAME)
    print(f"Model save to {MODEL_PATH + MODEL_NAME}.")


if __name__ == '__main__':
    main()
