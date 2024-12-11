from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from pathlib import Path
from torch import tensor

#############
# Problem 2 #
#############


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
        path = self.paths[index].as_posix()
        image = read_image(path).squeeze(dim=0) / 255
        label = int(path.split('/')[-2])

        return image, tensor(label)


def make_loader(subfolder: str):
    dataset = MNIST_dataset(f"mnist_png/{subfolder}")
    return DataLoader(dataset, batch_size=64, shuffle=True)


###########
# Program #
###########

def main():
    train = make_loader('train')
    val = make_loader('valid')


if __name__ == "__main__":
    main()
