import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, x_data: np.array, y_data: np.array, sequence_length: int):
        """
        Docstring for __init__

        :param self: Description
        :param x_data: tokens in our dataset
        :type x_data: np.array of shape (total_num_tokens, elements_in_universe)
        :param y_data: targets for our transformer
        :type y_data: np.array of shape (min(total_num_tokens - sequence_length), 2)
        :param device: device to put the dataset on
        :type device: torch.device
        :param sequence_length: Description
        """
        super().__init__()
        self.x = x_data  # (T, N, F)
        self.y = y_data  # (T, N, output_classes)
        self.seq_len = sequence_length

        T, N, F = x_data.shape

        self.valid_indices = np.arange(T - sequence_length + 1)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.seq_len

        x_seq = self.x[start:end, :, :]  # (seq_len, N, F)
        y_seq = self.y[start:end, :]  # (seq_len, N)

        return x_seq, y_seq
