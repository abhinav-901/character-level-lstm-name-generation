from src import from_configure
from tensorboardX import SummaryWriter
import torch
import random
from src.rnn_arch import RNN

# initial string for generation of next chars
initial_string = from_configure.START_STRING_FOR_PREDICTION


class Train():
    """ Training class
    """
    def __init__(self, unique_char: str, device, n_char: int, all_data: str):
        self.chunk_len = from_configure.SEQUENCE_LENGTH
        self.num_epochs = from_configure.NUM_EPOCH
        self.batch_size = from_configure.BATCH_SIZE
        self.print_every = from_configure.PRINT_EVERY
        self.n_hidden = from_configure.HIDDEN_SIZE
        self.n_layers = from_configure.N_LAYER
        self.lr = from_configure.LR
        self.drop_rate = from_configure.DROPOUT_RATE
        self.all_chars = unique_char
        self.device = device
        self.n_char = n_char
        self.all_data = all_data

    def char_tensor(self, text: str) -> torch.tensor:
        """ fucntion will map each character in string to a tensor

        Args:
            text (str): input text

        Returns:
            torch.tensor: encoded tensor for text
        """
        tensor = torch.zeros(len(text)).long()
        for index_ in range(len(text)):
            tensor[index_] = self.all_chars.index(text[index_])
        return tensor

    def get_random_batches(self) -> tuple:
        """ create random batches

        Returns:
            tuple: tuple contatining random batch
        """
        start_idx = random.randint(0, len(self.all_data) - self.chunk_len)
        end_idx = start_idx + self.chunk_len + 1
        text_str = self.all_data[start_idx:end_idx]
        input_text = torch.zeros(self.batch_size, self.chunk_len)
        target_text = torch.zeros(self.batch_size, self.chunk_len)
        for i in range(self.batch_size):
            input_text[i, :] = self.char_tensor(text_str[:-1])
            target_text[i, :] = self.char_tensor(text_str[1:])
        return input_text.long(), target_text.long()

    def generate(self,
                 initial_string: str = initial_string,
                 predict_len: int = 100,
                 tempreature: float = from_configure.TEMPREATURE) -> str:
        """ generate prediction upto n chars for given initial string

        Args:
            initial_string (str, optional): initial string for prediction.
            predict_len (int, optional): max length for text generation. Defaults to 100.
            tempreature (float, optional): hyperparameter to caliberate network.

        Returns:
            str: generated string
        """

        hidden, cell = self.rnn.init_hidden(self.batch_size)
        initial_inp = self.char_tensor(initial_string)
        predicted_str = initial_string
        for i in range(len(initial_string) - 1):
            _, (hidden,
                cell) = self.rnn(initial_inp[i].view(1).to(self.device),
                                 hidden, cell)
        last_char = initial_inp[-1]
        for i in range(predict_len):
            out, (hidden, cell) = self.rnn(
                last_char.view(1).to(self.device), hidden, cell)
            out_dist = out.view(-1).div(tempreature).exp()
            top_chars = torch.multinomial(out_dist, 1)[0]
            predicted_char = self.all_chars[top_chars]
            predicted_str += predicted_char
            last_char = self.char_tensor(predicted_char)
        return predicted_str

    def train(self):
        """ method to train the network
        """
        self.rnn = RNN(self.n_char, self.n_layers, self.n_hidden,
                       self.drop_rate, self.n_char,
                       self.device).to(self.device)
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        criterian = self.rnn.criterian
        writer = SummaryWriter('./out/logdir')
        for epoch in range(1, self.num_epochs):
            data, label = self.get_random_batches()
            data, label = data.to(self.device), label.to(self.device)
            hidden, cell = self.rnn.init_hidden(self.batch_size)
            loss = 0
            self.rnn.zero_grad()
            for char in range(0, self.chunk_len):
                output, (hidden,
                         cell) = self.rnn.forward(data[:, char], hidden, cell)
                loss += criterian(output, label[:, char])
            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len
            if epoch % self.print_every == 0:
                print(
                    f"Training loss for epoch : {epoch}/{self.num_epochs} --> {loss}"
                )
                print(self.generate())
            writer.add_scalar('training_loss', loss, global_step=epoch)
