import torch.nn as nn
import numpy as np
import torch
from src import from_configure


class RNN(nn.Module):
    """ custom RNN class subclassed from nn.Module"""
    def __init__(self, input_size, n_layers, n_hidden, drop_rate, out_size,
                 device):
        super(RNN, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.drop_rate = drop_rate
        self.out_size = out_size
        self.input_size = input_size
        self.num_directions = from_configure.NUM_DIRECTIONS
        self.is_bidirectional = from_configure.BI_DR
        self.device = device

        # dropout layer
        self.dropout = nn.Dropout(self.drop_rate)

        # embedding layer
        self.embed = nn.Embedding(self.input_size, self.n_hidden)

        # lstm layer
        self.lstm = nn.LSTM(self.input_size,
                            self.n_hidden,
                            self.n_layers,
                            dropout=self.drop_rate,
                            batch_first=True,
                            bidirectional=self.is_bidirectional)

        # fully connected layer
        self.fc1 = nn.Linear(self.n_hidden * self.num_directions, 258)
        self.fc2 = nn.Linear(258, self.out_size)

        # loss for the network
        self.criterian = nn.CrossEntropyLoss()

    def forward(self, x: torch.tensor, hidden: torch.tensor,
                cell: torch.tensor) -> tuple:
        """ forward pas of the network

        Args:
            x (torch.tensor): input tensor
            hidden (torch.tensor): hidden state
            cell (torch.tensor): cell state

        Returns:
            tuple: (out, hidden, cell)
        """
        out: torch.tensor = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = self.dropout(out)
        out = self.fc1(out.view(out.shape[0], -1))
        out = self.dropout(out)
        out = self.fc2(out)
        return out, (hidden, cell)

    def init_hidden(self, batch_size: int) -> tuple:
        """ initialize hidden and cell state of the lstm

        Args:
            batch_size (int): samples of sequence network will see in one epoch

        Returns:
            tuple: [tensor, tensor]
        """
        hidden = torch.zeros(self.n_layers * self.num_directions, batch_size,
                             self.n_hidden).to(self.device)
        cell = torch.zeros(self.n_layers * self.num_directions, batch_size,
                           self.n_hidden).to(self.device)
        return hidden, cell
