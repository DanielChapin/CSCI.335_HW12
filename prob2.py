from hw12 import MNIST_dataset

dataset = MNIST_dataset("./mnist_png")

print('\n'.join(map(lambda x: str(dataset[x]), range(10))))
