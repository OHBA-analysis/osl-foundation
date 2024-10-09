import numpy as np

from osl_dynamics import data as osld_data


class Data(osld_data.Data):
    def concatenate_channels(self):
        """Concatenate channels of the data arrays."""
        self.original_n_channels = self.n_channels
        # (n_samples, n_channels) -> (n_samples * n_channels, 1)
        self.arrays = [np.reshape(x, (-1, 1), order="F") for x in self.arrays]

    def split_channels(self):
        """Split the sequences back to different channels."""
        self.arrays = [
            np.reshape(x, (-1, self.original_n_channels), order="F")
            for x in self.arrays
        ]
        self.original_n_channels = None
