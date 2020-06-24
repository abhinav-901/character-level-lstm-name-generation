import torch
from string import printable
from tensorboardX import SummaryWriter
from src.train_model import Train


def main():
    # select GPU if available else go with CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # reading name text from file
    data_file = open('./data/indian_names.txt', 'r+').read()

    # all unique characters
    all_chars = printable

    # number of output class
    n_chars = len(all_chars)
    train_model = Train(all_chars, device, n_chars, data_file)
    train_model.train()


if __name__ == '__main__':
    main()