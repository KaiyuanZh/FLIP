import torch

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

AGGR_MEAN = "mean"
MAX_UPDATE_NORM = 1000
patience_iter = 20

TYPE_CIFAR = "cifar"
TYPE_MNIST = "mnist"
TYPE_FASHION_MNIST = "fashion_mnist"
