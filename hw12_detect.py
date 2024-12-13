import os
import re
from sys import argv
from hw12 import MyConvModel
from torch import argmax, load
from torchvision.io import read_image


#############
# Problem 5 #
#############

# Gets all the paths, times, and accuracies of the files in the weights directory.
def weights_paths(directory: str = "weights") -> list[(str, int, float)]:
    file_format = re.compile(r"^weights_(?P<time>\d+)_(?P<acc>\d+).pth$")
    files = os.listdir(directory)
    matches = map(lambda path:
                  re.match(file_format, os.path.basename(path)),
                  files)
    return [(os.path.join(directory, rem.group(0)), int(rem.group("time")), int(rem.group("acc")) / 1e6)
            for rem in matches if rem != None]


def best_weights_path():
    return max(weights_paths(), key=lambda x: x[2])[0]


def grab_image(path):
    image = read_image(path)
    grayscale = image.squeeze(dim=0) / 255
    return grayscale.unsqueeze(dim=0)


def main():
    if len(argv) == 1:
        print(f"Usage: py {argv[0]} <filepaths to predict>")
        return

    model = MyConvModel()
    path = best_weights_path()
    print(f"Loading model from {path}")
    model.load_state_dict(load(path, weights_only=True))
    model.eval()

    image_paths = argv[1:]

    for image in map(grab_image, image_paths):
        pred = argmax(model.forward(image)).item()
        print(pred)


if __name__ == "__main__":
    main()

# Analysis of digits from HW 11:
# $ python .\hw12_detect.py .\my_images\4.png .\my_images\7.png .\my_images\3.png
# 4
# 7
# 2
#
# The model only got the off-center 3 wrong.
# The placement of the digit is indeed important to the result.
# If I take that same 3 and shift it into the center, it is able to identify it correctly.
# I through in the center dash in the 7 to try to trick it, but it still identified it correctly.
# This makes me incredible curious as to how OCR works seeing as it's able to identify
# characters quite accurately in far worse conditions.
# After the model loads it does run practically instantly, so the performance of the
# forward propagation is quite good speed wise (albeit this is a smaller model).
#
# I also drew an 'f' just to see how it would classify it and it guesses 7, which seems
# like a pretty reasonable guess.
