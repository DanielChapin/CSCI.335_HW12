import os
import re
from sys import argv
from hw12 import MyConvModel, grab_image
from torch import argmax, load
from pathlib import Path


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


def main():
    if len(argv) == 1:
        print(f"Usage: py {argv[0]} <filepaths to predict>")
        return

    model = MyConvModel()
    path = best_weights_path()
    print(f"Loading model from {path}")
    model.load_state_dict(load(path))
    model.eval()

    image_paths = argv[1:]
    images = map(lambda path: grab_image(Path(path)), image_paths)

    for image, y in images:
        pred = argmax(model.forward(image)).item()
        print(pred, end="")
        if pred != y:
            print(f" (Expected {y})", end="")
        print()


if __name__ == "__main__":
    main()
