import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
        self,
        x_data: list[np.ndarray],
        y_data: np.array,
        sequence_length: int,
        image_height=16,
    ):
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
        self.x = x_data  # list[...(T, N, F)]
        self.y = y_data  # (T, N, output_classes)
        self.seq_len = sequence_length
        self.image_height = image_height

        # self.T, self.N, self.F = x_data.shape

        total_dates = x_data[0].shape[0]

        self.valid_indices = np.arange(total_dates - sequence_length + 1)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.seq_len

        out = []
        y_seq: np.ndarray = self.y[start:end, :]  # (seq_len, N)

        second = False
        for partial_sequence in self.x:
            x_seq: np.ndarray = partial_sequence[start:end, :, :]  # (seq_len, N, F)
            T, N, F = x_seq.shape

            # print("x_seq_shape", x_seq.shape) # (T, N, F)
            x_seq = x_seq.transpose(1, 0, 2)  # (N, T, F)
            min_per = np.min(x_seq.reshape(N, -1), axis=-1)  # (N, )
            x_seq = x_seq.reshape(N, -1) - min_per.reshape(N, 1)  # (N, T*F)
            remaining_max = np.max(x_seq, axis=-1)  # (N, )
            # avoid division by zero: replace zeros with 1 (x_seq will be all zeros there anyway)
            remaining_max_safe = np.where(remaining_max == 0, 1, remaining_max)
            x_seq = np.round(
                (x_seq / remaining_max_safe.reshape(N, 1))
                * (self.image_height - 1)  # because 0 index
            )  # (N, seq_len*F), every value is integer [0, self.image_height]
            # print(x_seq.shape, N, T, F)
            x_seq = x_seq.reshape(N, T, F)
            if second:
                print(x_seq.reshape(N, -1))
            second = True
            one_hot = np.eye(self.image_height, dtype=int)[
                x_seq.flatten().astype(np.int32)
            ]  # (N, T, F, image_height)
            one_hot = one_hot.reshape(N, T, F, self.image_height)

            # print(one_hot)
            # print("one hot shape", one_hot.shape)
            [out.append(chunk) for chunk in np.split(one_hot, F, axis=2)]

        # [print(t.shape) for t in out]
        x_seq = np.stack(out, axis=2)  # (N, T, xxx, image_height)
        N, T, total_features, _, im_height = x_seq.shape
        x_seq = x_seq.reshape(N, T, total_features, im_height)
        # print(x_seq.shape)

        return x_seq, y_seq
