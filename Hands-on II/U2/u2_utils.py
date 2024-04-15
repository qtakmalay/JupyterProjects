# -*- coding: utf-8 -*-
"""
Authors: B. Schäfl, S. Lehner, J. Brandstetter, A. Schörgenhumer
Date: 21-03-2023

This file is part of the "Hands-on AI II" lecture material. The following copyright statement applies
to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for personal and non-commercial
educational use only. Any reproduction of this manuscript, no matter whether as a whole or in parts, no matter whether
in printed or in electronic form, requires explicit prior acceptance of the authors.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import sys
import torch
import random

from distutils.version import LooseVersion
from IPython.core.display import HTML
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from typing import Optional, Sequence, Tuple, Union

# https://stackoverflow.com/a/69692664/8176827
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


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
    torch_check = '(\u2713)' if LooseVersion(torch.__version__) >= LooseVersion('1.6.0') else '(\u2717)'
    sklearn_check = '(\u2713)' if LooseVersion(sklearn.__version__) >= LooseVersion('1.2') else '(\u2717)'
    matplotlib_check = '(\u2713)' if LooseVersion(matplotlib.__version__) >= LooseVersion('3.2.0') else '(\u2717)'
    seaborn_check = '(\u2713)' if LooseVersion(sns.__version__) >= LooseVersion('0.10.0') else '(\u2717)'
    print(f'Installed Python version: {sys.version_info.major}.{sys.version_info.minor} {python_check}')
    print(f'Installed numpy version: {np.__version__} {numpy_check}')
    print(f'Installed pandas version: {pd.__version__} {pandas_check}')
    print(f'Installed PyTorch version: {torch.__version__} {torch_check}')
    print(f'Installed scikit-learn version: {sklearn.__version__} {sklearn_check}')
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


def load_mnist() -> pd.DataFrame:
    """
    Load MNIST data set [1].

    [1] LeCun, Y., 1998. The MNIST database of handwritten digits. http://yann.lecun.com/exdb/mnist/.

    :return: MNIST data set
    """
    mnist_data = datasets.fetch_openml(name=r'mnist_784', as_frame=True, parser="auto")
    data = pd.DataFrame(mnist_data["data"]).astype(np.float32)
    data["digit"] = mnist_data["target"].astype(int)
    return data


def load_fashion_mnist() -> pd.DataFrame:
    """
    Load Fashion-MNIST data set [1].

    [1] Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
        Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747

    :return: Fashion-MNIST data set
    """
    fashion_mnist_data = datasets.fetch_openml(name=r'Fashion-MNIST', as_frame=True, parser="auto")
    data = fashion_mnist_data["data"].astype(np.float32)
    data["item_type"] = fashion_mnist_data["target"].astype(int)
    return data


def split_data(data: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data set into training and testing subsets.

    :param data: data set to split
    :param test_size: relative size of the test subset
    :return: training as well as testing subsets
    """
    assert (data is not None) and (type(data) == pd.DataFrame)
    assert (test_size is not None) and (type(test_size) == float) and (0 < test_size < 1)
    return train_test_split(data, test_size=test_size)


def apply_pca(data: pd.DataFrame, n_components: Optional[int] = None, target_column: Optional[str] = None,
              components: Optional[PCA] = None, return_components: bool = False
              ) -> Union[Tuple[pd.DataFrame, PCA], pd.DataFrame]:
    """
    Apply principal component analysis (PCA) on specified dataset and down-project project data accordingly.

    :param data: dataset to down-project
    :param n_components: amount of (top) principal components involved in down-projection
    :param target_column: if specified, append target column to resulting, down-projected data set
    :param return_components: return principal components in addition of down-projected data set
    :param components: use these principal components instead of freshly computing them
    :return: down-projected data set and optionally principal components
    """
    assert type(data) == pd.DataFrame
    assert ((n_components is None) and (components is not None)) or (type(n_components) == int) and (n_components >= 1)
    assert ((type(target_column) == str) and (target_column in data)) or (target_column is None)
    assert (components is None) or (type(components) == PCA)
    assert type(return_components) == bool
    
    if target_column is not None:
        target_data = data[target_column]
        data = data.drop(columns=target_column)
    print(target_data)
    if components is None:
        components = PCA(n_components=n_components).fit(data)
    projected_data = pd.DataFrame(components.transform(data), index=data.index)
    if target_column is not None:
        projected_data[target_column] = target_data
    
    return (projected_data, components) if return_components else projected_data


def apply_tsne(n_components: int, data: pd.DataFrame, target_column: Optional[str] = None,
               perplexity: float = 10.0) -> pd.DataFrame:
    """
    Apply t-distributed stochastic neighbor embedding (t-SNE) on specified dataset and down-project data accordingly.

    :param n_components: dimensionality of the embedding space
    :param data: dataset to down-project
    :param target_column: if specified, append target column to resulting, down-projected dataset
    :param perplexity: this term is closely related to the number of nearest neighbors to consider
    :return: down-projected dataset
    """
    assert (type(n_components) == int) and (n_components >= 1)
    assert type(data) == pd.DataFrame
    assert ((type(target_column) == str) and (target_column in data)) or (target_column is None)
    assert (type(perplexity) == float) or (type(perplexity) == int)
    if target_column is not None:
        projected_data = pd.DataFrame(TSNE(n_components=n_components, perplexity=float(perplexity), learning_rate=200,
                                           init="random").fit_transform(data.drop(columns=target_column)),
                                      index=data.index)
        projected_data[target_column] = data[target_column]
    else:
        projected_data = pd.DataFrame(TSNE(n_components=n_components, perplexity=float(perplexity), learning_rate=200,
                                           init="random").fit_transform(data), index=data.index)
    return projected_data


def plot_points_2d(data: pd.DataFrame, target_column: Optional[str] = None, targets: Sequence = None,
                   legend: bool = True, multi_color_palette: str = "husl", **kwargs) -> None:
    """
    Visualize data points in a two-dimensional plot, optionally color-coding according to ``target_column``.

    :param data: dataset to visualize
    :param target_column: optional target column to be used for color-coding
    :param targets: sequence of target labels if not contained in ``data`` (via ``target_column``)
    :param legend: flag for displaying a legend
    :param multi_color_palette: Seaborn color palette to use when > 10 targets are plotted
    :param kwargs: optional keyword arguments passed to ``plt.subplots``
    """
    assert (type(data) == pd.DataFrame) and (data.shape[1] in [2, 3])
    assert not (
                target_column is not None and targets is not None), "can only specify either 'target_column' or 'targets"
    assert (target_column is None) or ((data.shape[1] == 3) and (data.columns[2] == target_column))
    assert targets is None or len(targets) == len(data)
    fig, ax = plt.subplots(**kwargs)
    color_targets = None
    if target_column is not None:
        color_targets = data[target_column]
    elif targets is not None:
        color_targets = targets
    color_palette = None
    if color_targets is not None:
        n_colors = len(set(color_targets))
        palette = "muted" if n_colors <= 10 else multi_color_palette
        color_palette = sns.color_palette(palette=palette, n_colors=n_colors)
    legend = "auto" if legend else False
    sns.scatterplot(x=data[0], y=data[1], hue=color_targets, ax=ax, palette=color_palette, legend=legend)
    plt.tight_layout()
    plt.show()


def train_network(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer, device: torch.device = r'cpu') -> None:
    """
    Train specified network for one epoch on specified data loader.

    :param model: network to train
    :param data_loader: data loader to be trained on
    :param optimizer: optimizer used to train network
    :param device: device on which to train network
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for batch_index, (data, target) in enumerate(data_loader):
        data, target = data.float().to(device), target.long().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test_network(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                 device: torch.device = r'cpu') -> Tuple[float, float]:
    """
    Test specified network on specified data loader.

    :param model: network to test on
    :param data_loader: data loader to be tested on
    :param device: device on which to test network
    :return: cross-entropy loss as well as accuracy
    """
    model.eval()
    loss = 0.0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.float().to(device), target.long().to(device)
            output = model(data)
            loss += float(criterion(output, target).item())
            pred = output.max(1, keepdim=True)[1]
            correct += int(pred.eq(target.view_as(pred)).sum().item())
    
    return loss / len(data_loader.dataset), correct / len(data_loader.dataset)
