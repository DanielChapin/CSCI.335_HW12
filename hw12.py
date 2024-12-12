from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from pathlib import Path
from torch import tensor
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch
from time import time

#############
# Problem 2 #
#############


def grab_image(path: str):
    path = path.as_posix()
    image = read_image(path).squeeze(dim=0) / 255
    label = int(path.split('/')[-2])

    return image.unsqueeze(dim=0), tensor(label)


class MNIST_dataset(Dataset):
    datapath: Path
    paths: list[Path]

    def __init__(self, datapath):
        super().__init__()
        self.datapath = Path(datapath)
        self.paths = list(self.datapath.rglob("*.png"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return grab_image(self.paths[index])


def make_loader(subfolder: str):
    dataset = MNIST_dataset(f"mnist_png/{subfolder}")
    return DataLoader(dataset, batch_size=64, shuffle=True)

#############
# Problem 3 #
#############


class MyConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolutional layer
        # Only one input channel because we're working with black and white images
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        # Subsampling layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Second subsampling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Output layer
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool1(x)

        x = nn.ReLU()(self.conv2(x))
        x = self.pool2(x)

        # Need to flatten data for linear layer
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc1(x)

        return x


#############
# Problem 4 #
#############

def main():
    # Grabbing data
    train = make_loader('train')
    val = make_loader('valid')

    # Model training
    model = MyConvModel()
    critereion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    num_epochs = 10
    accuracies = []
    for epoch in range(num_epochs):
        print(f"Begin epoch {epoch}")
        print("Training...")
        model.train()
        for x, y in train:
            y_pred = model.forward(x)
            loss = critereion(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        print("Validating...")
        with torch.no_grad():
            acc_val = []
            for x, y in val:
                y_pred = model.forward(x)
                acc = accuracy_score(y, torch.argmax(
                    y_pred, dim=-1, keepdim=True))
                acc_val.append(acc)
            acc = sum(acc_val) / len(acc_val)
            accuracies.append(acc)
            print(f"Accuracy of {acc}")

    # Saving the weights
    # File format: weights_<timestamp>_<accuracy>.pt
    filepath = f"weights/weights_{round(time())}_{round(accuracies[-1] * 1e6)}.pth"
    torch.save(model.state_dict(), filepath)
    print(f"Weights saved to {filepath}")


if __name__ == "__main__":
    main()
