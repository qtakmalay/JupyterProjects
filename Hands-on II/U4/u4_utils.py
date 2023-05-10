# -*- coding: utf-8 -*-
"""
Authors: B. Schäfl, S. Lehner, J. Schimunek, J. Brandstetter, A. Schörgenhumer
Date: 02-05-2023

This file is part of the "Hands-on AI II" lecture material. The following copyright statement applies
to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for personal and non-commercial
educational use only. Any reproduction of this manuscript, no matter whether as a whole or in parts, no matter whether
in printed or in electronic form, requires explicit prior acceptance of the authors.
"""
import random
import sys
from copy import deepcopy
from distutils.version import LooseVersion
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.core.display import HTML
from matplotlib.colors import ListedColormap


def setup_jupyter() -> HTML:
    """
    Setup Jupyter notebook. Warning: this may affect all Jupyter notebooks running on the same Jupyter server.

    :return: HTML instance comprising the modified Jupyter attributes
    """
    return HTML(r"""
    <style>
        .output_png {
            display: table-cell;
            text-align: center;
            vertical-align: middle;
        }
        .jp-RenderedImage {
            display: table-cell;
            text-align: center;
            vertical-align: middle;
        }
    </style>
    <p>Setting up notebook ... finished.</p>
    """)


# noinspection PyUnresolvedReferences
def check_module_versions() -> None:
    """
    Check Python version as well as versions of recommended (partly required) modules.
    """
    python_check = '(\u2713)' if sys.version_info >= (3, 8) else '(\u2717)'
    numpy_check = '(\u2713)' if LooseVersion(np.__version__) >= LooseVersion('1.18') else '(\u2717)'
    pandas_check = '(\u2713)' if LooseVersion(pd.__version__) >= LooseVersion('1.0') else '(\u2717)'
    torch_check = '(\u2713)' if LooseVersion(torch.__version__) >= LooseVersion('1.6') else '(\u2717)'
    matplotlib_check = '(\u2713)' if LooseVersion(matplotlib.__version__) >= LooseVersion('3.2.0') else '(\u2717)'
    seaborn_check = '(\u2713)' if LooseVersion(sns.__version__) >= LooseVersion('0.10.0') else '(\u2717)'
    print(f'Installed Python version: {sys.version_info.major}.{sys.version_info.minor} {python_check}')
    print(f'Installed numpy version: {np.__version__} {numpy_check}')
    print(f'Installed pandas version: {pd.__version__} {pandas_check}')
    print(f'Installed PyTorch version: {torch.__version__} {torch_check}')
    print(f'Installed matplotlib version: {matplotlib.__version__} {matplotlib_check}')
    print(f'Installed seaborn version: {sns.__version__} {seaborn_check}')


def set_seed(seed: int = 42) -> None:
    """
    Set seed for all underlying (pseudo) random number sources.

    :param seed: seed to be used
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_heatmap(data: pd.DataFrame, prefix_mask: bool = False,
                 prefix_index: int = 2, prefix_length: int = 2, **kwargs) -> None:
    """
    Visualize the specified data set in a heatmap, with optional prefix-masking.
    
    :param data: data set for which to visualize the heatmap
    :param prefix_mask: highlight prefix including consecutive data element
    :param prefix_index: index to be interpreted as prefix-specific
    :param prefix_length: length of the prefix
    """

    # Optionally compute mask w.r.t. to prefix (as e.g. used in visualizing latch sequence data).
    mask = False
    if prefix_mask:
        mask = ~(data.transpose().rolling(window=prefix_length + 1, min_periods=1).apply(
            lambda window: all((
                len(window) == prefix_length + 1,
                (window.to_numpy().astype(np.int)[:prefix_length] == (prefix_index, prefix_index)).all()
            ))).transpose().astype(bool) | (data == prefix_index))

    # Create heatmap of specified data with optional prefix masking.
    fig, ax = plt.subplots(**kwargs)
    num_labels = np.unique(data.to_numpy().flatten()).shape[0]
    cmap = ListedColormap(sns.color_palette('Spectral', num_labels))
    ax = sns.heatmap(data, cmap=cmap, ax=ax, mask=mask, vmin=0, vmax=num_labels - 1)
    ax.grid(False)

    # Adapt color bar to specified data.
    color_bar = ax.collections[0].colorbar
    color_bar.set_ticks(color_bar._values)
    color_bar.set_ticklabels([chr(_) for _ in range(65, 65 + num_labels)])


def train_network(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer, device: torch.device = 'cpu') -> None:
    """
    Train specified network for one epoch on specified data loader.

    :param model: network to train
    :param data_loader: data loader to be trained on
    :param optimizer: optimizer used to train network
    :param device: device on which to train network
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for data, target in data_loader:
        data, target = data.float().to(device), target.long().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test_network(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                 device: torch.device = 'cpu') -> Tuple[float, float]:
    """
    Test specified network on specified data loader.

    :param model: network to test on
    :param data_loader: data loader to be tested on
    :param device: device on which to test network
    :return: cross-entropy loss as well as accuracy
    """
    model.eval()
    loss, num_correct, num_samples = 0.0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.float().to(device), target.long().to(device)
            output = model(data)
            loss += float(criterion(output, target).item())
            pred = output.argmax(dim=1).view(-1).long()
            num_correct += int((pred == target.view(-1)).sum().item())
            num_samples += pred.shape[0]
    return loss / num_samples, num_correct / num_samples


class LatchSequenceSet(torch.utils.data.Dataset):
    """
    Latch data set comprising patterns as one-hot encoded instances.
    """

    def __init__(self, num_samples: int, num_instances: int = 20, num_characters: int = 6,
                 num_targets: int = 2, dtype: torch.dtype = torch.float32, seed: int = 42):
        """
        Create new latch sequence data set conforming to the specified properties.
        
        :param num_samples: amount of samples
        :param num_instances: amount of instances per sample
        :param num_characters: amount of different characters
        :param dtype: data type of samples
        :param seed: random seed used to generate the samples of the data set
        """
        super(LatchSequenceSet, self).__init__()
        assert (type(num_samples) == int) and (num_samples >= 1), '"num_samples" must be a positive integer!'
        assert (type(num_instances) == int) and (num_instances >= 3), '"num_instances" must be at least 3!'
        assert (type(num_targets) == int) and (num_targets >= 2), '"num_targets" must be at least 2!'
        assert (type(num_characters) == int) and (num_characters >= (num_targets + 1)), '"num_characters" must be at least "num_targets + 1"!'
        assert type(seed) == int, '"seed" must be an integer!'

        self.__num_samples = num_samples
        self.__num_instances = num_instances
        self.__num_characters = num_characters
        self.__num_targets = num_targets
        self.__dtype = dtype
        self.__seed = seed
        self.__data, self.__targets, self.__positions = self._generate_latch_sequences()

    def __len__(self) -> int:
        """
        Fetch amount of samples.
        
        :return: amount of samples
        """
        return self.__num_samples

    def __getitem__(self, item_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch specific sample as 2-tuple: [0] = data, [1] = target
        
        :param item_index: specific sample to fetch
        :return: specific sample as tuple of tensors
        """
        return (self.__data[item_index].to(dtype=self.__dtype),
                self.__targets[item_index].to(dtype=self.__dtype))

    def _generate_latch_sequences(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate underlying latch sequence data set.
        
        :return: tuple containing generated data and targets
        """
        set_seed(self.__seed)

        # Generate raw data tensor.
        generated_data = [_ for _ in range(self.__num_characters) if _ != self.__num_targets]
        generated_data = np.random.choice(generated_data, size=(self.__num_samples, self.__num_instances))
        generated_data = torch.as_tensor(generated_data).to(dtype=torch.long)
        
        # Generate signal tensor, including emplacement positions.
        affected_samples = torch.arange(self.__num_samples)
        generated_signal_positions = torch.randint(
            low=2, high=self.__num_instances, size=(self.__num_samples,))
        generated_signal = torch.randint(
            low=0, high=self.__num_targets, size=(self.__num_samples,),
            dtype=generated_data.dtype)
        
        # Combine data as well as signal tensors.
        generated_data[affected_samples, generated_signal_positions] = generated_signal
        for prefix_position in range(1, 3):
            adapted_signal_positions = generated_signal_positions - prefix_position
            generated_data[affected_samples, adapted_signal_positions] = self.__num_targets
        generated_data = torch.nn.functional.one_hot(
            input=generated_data, num_classes=self.__num_characters)

        return generated_data, generated_signal, generated_signal_positions

    @property
    def num_samples(self) -> int:
        return self.__num_samples

    @property
    def num_instances(self) -> int:
        return self.__num_instances

    @property
    def num_characters(self) -> int:
        return self.__num_characters
    
    @property
    def num_targets(self) -> int:
        return self.__num_targets

    @property
    def initial_seed(self) -> int:
        return self.__seed

    @property
    def targets(self) -> torch.Tensor:
        return self.__targets.clone()
    
    @property
    def positions(self) -> torch.Tensor:
        return self.__positions.clone()
    
    def resort(self, reverse: bool = False) -> 'LatchSequenceSet':
        """
        Returns a sorted copy of the latch data where the class defining subsequence
        is always at the start of the sequence (`reverse = False`) or at the back of
        the sequence (`reverse = True`).
        
        :param reverse: True if the class defining subsequence should be at the back
            of the sequence, or False if it should be at the start of the sequence
        :return: A sorted copy of the latch data
        """
        sorted_data = deepcopy(self)
        for sequence, position in zip(sorted_data.__data, sorted_data.__positions):
            if reverse:
                from_slice = slice(max(len(sequence) - 3, position + 1), None)
                to_slice = slice(position - 2, position - 2 + min(3, len(sequence) - 1 - position))
                new_target_pos_slice = slice(-3, None)
            else:
                from_slice = slice(min(3, position - 2))
                to_slice = slice(position - min(2, position - 3), position + 1)
                new_target_pos_slice = slice(3)
            triple_source = sequence[from_slice].clone()
            triple_target = sequence[position - 2:position + 1].clone()
            sequence[new_target_pos_slice] = triple_target
            sequence[to_slice] = triple_source
        sorted_data.__positions = torch.full_like(sorted_data.__positions, self.num_instances - 1 if reverse else 2)
        return sorted_data
