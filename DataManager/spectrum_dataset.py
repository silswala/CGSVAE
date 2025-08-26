from torch.utils.data import Dataset
import torch


class SpectraDataset(Dataset):
    """Dataset class for loading and managing spectral data and labels.

    Args:
        spectra_list (list or numpy.ndarray): List or array of spectra (data). Each element should be a spectrum.
        labels_list (list or numpy.ndarray): List or array of labels corresponding to the spectra.  Must be the same length as `spectra_list`.

    Raises:
        AssertionError: If `spectra_list` and `labels_list` have different lengths.

    Attributes:
        spectra (list): List of spectra (torch.Tensor).
        labels (list): List of labels (torch.Tensor).
    """

    def __init__(self, spectra_list, labels_list):
        """Initializes the dataset.

        Converts the input lists/arrays to lists of torch.Tensors.
        """
        assert len(spectra_list) == len(labels_list), "Spectra and labels must have the same length."

        self.spectra = []
        self.labels = []

        for spectrum in spectra_list:
            self.spectra.append(torch.tensor(spectrum, dtype=torch.float32))
        for label in labels_list:
            self.labels.append(torch.tensor(label, dtype=torch.float32))

    def __len__(self):
        """Returns the total number of samples (spectra).

        Returns:
            int: The number of spectra in the dataset.
        """
        return len(self.spectra)

    def __getitem__(self, idx):
        """Retrieves the spectrum and corresponding label at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing:
                - spectrum (torch.Tensor): The spectrum data.
                - label (torch.Tensor): The corresponding label.
        """
        spectrum = self.spectra[idx]
        label = self.labels[idx]
        return spectrum, label
