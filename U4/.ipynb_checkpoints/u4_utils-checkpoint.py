# -*- coding: utf-8 -*-
"""
Authors: Brandstetter, Schäfl, Mitterecker, Ramsauer, Rumetshofer, Schörgenhumer
Date: 21-11-2022

This file is part of the "Hands on AI I" lecture material. The following copyright statement applies
to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for personal and non-commercial
educational use only. Any reproduction of this manuscript, no matter whether as a whole or in parts, no matter whether
in printed or in electronic form, requires explicit prior acceptance of the authors.
"""
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import sklearn
import sys
import torch
import torchvision

from distutils.version import LooseVersion
from IPython.core.display import HTML
from math import prod
from pathlib import Path
from scipy.special import softmax
from torch.utils.data import DataLoader
from typing import Callable, Sequence, Tuple, Union


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


def check_module_versions() -> None:
    """
    Check Python version as well as versions of recommended (partly required) modules.
    """
    python_check = '(\u2713)' if sys.version_info >= (3, 8) else '(\u2717)'
    numpy_check = '(\u2713)' if LooseVersion(np.__version__) >= LooseVersion(r'1.18') else '(\u2717)'
    pandas_check = '(\u2713)' if LooseVersion(pd.__version__) >= LooseVersion(r'1.0') else '(\u2717)'
    sklearn_check = '(\u2713)' if LooseVersion(sklearn.__version__) >= LooseVersion(r'0.23') else '(\u2717)'
    matplotlib_check = '(\u2713)' if LooseVersion(matplotlib.__version__) >= LooseVersion(r'3.2.0') else '(\u2717)'
    seaborn_check = '(\u2713)' if LooseVersion(sns.__version__) >= LooseVersion(r'0.10.0') else '(\u2717)'
    scipy_check = '(\u2713)' if LooseVersion(scipy.__version__) >= LooseVersion(r'1.5.0') else '(\u2717)'
    torch_check = '(\u2713)' if LooseVersion(torch.__version__) >= LooseVersion(r'1.6.0') else '(\u2717)'
    print(f'Installed Python version: {sys.version_info.major}.{sys.version_info.minor} {python_check}')
    print(f'Installed numpy version: {np.__version__} {numpy_check}')
    print(f'Installed pandas version: {pd.__version__} {pandas_check}')
    print(f'Installed scikit-learn version: {sklearn.__version__} {sklearn_check}')
    print(f'Installed matplotlib version: {matplotlib.__version__} {matplotlib_check}')
    print(f'Installed seaborn version: {sns.__version__} {seaborn_check}')
    print(f'Installed scipy version: {scipy.__version__} {scipy_check}')
    print(f'Installed torch version: {torch.__version__} {torch_check}')


def set_seed(seed: int = 42) -> None:
    """
    Set seed for all underlying (pseudo) random number sources.

    :param seed: Deed to be used.
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(num_pairs: int, variance: Union[int, float],
                coefficients: Sequence[Union[int, float]] = (0.241, 0.422)) -> pd.DataFrame:
    """
    Create dataset consisting of randomly generated (x, y) pairs.

    :param num_pairs: Amount of (x, y) pairs to generate.
    :param variance: Variance within (y,) w.r.t. defining function.
    :param coefficients: Coefficients of underlying polynomial regression model.
    :return: Dataset/data frame consisting of randomly generated (x, y) pairs.
    """
    assert num_pairs >= 1, 'At least one pair has to be generated.'
    assert variance >= 0, 'Variance has to be non-negative.'
    x, y = np.random.randn(num_pairs), np.zeros(num_pairs)
    for degree, coefficient in enumerate(coefficients):
        y += coefficient * x ** degree
    if variance > 0:
        y += variance * np.random.randn(num_pairs)
    return pd.DataFrame({"x": x, "y": y})


def get_dataset_unknown(num_pairs: int, variance: Union[int, float], degree_max: int = 6) -> pd.DataFrame:
    """
    Create dataset consisting of randomly generated (x, y) pairs using random coefficients.

    :param num_pairs: Amount of (x, y) pairs to generate.
    :param variance: Variance within (y,) w.r.t. defining function.
    :param degree_max: Upper bound (inclusive) of random polynomial degree.
    :return: Dataset/data frame consisting of randomly generated (x, y) pairs.
    """
    assert degree_max >= 1, 'At least a degree of 1 is required.'
    rnd_coefficients = np.random.randn(np.random.randint(low=2, high=degree_max + 2))
    return get_dataset(num_pairs=num_pairs, variance=variance, coefficients=rnd_coefficients)


def get_dataset_logistic(num_pairs: int) -> pd.DataFrame:
    """
    Create classification dataset consisting of randomly generated (x, y) pairs.

    :param num_pairs: Amount of (x, y) pairs to generate.
    :return: Dataset/data frame consisting of randomly generated (x, y) pairs.
    """
    assert num_pairs >= 1, 'At least one pair has to be generated.'
    x = np.random.rand(num_pairs)
    y = x.round().astype(int)
    return pd.DataFrame({"x": x, "y": y})


def get_dataset_from_csv(path: str, delimiter: str = ',', ignore_header: bool = False) -> pd.DataFrame:
    """
    Load data set from specified <*.csv> file.

    :param path: Path to <*.csv> file.
    :param delimiter: Delimiter used as separator in <*.csv> file.
    :param ignore_header: Whether to ignore the header and use a default header, where
        all columns except the last will be named "x0", "x1", etc. and the last column
        will be named "y".
    :return: Dataset/data frame consisting of data loaded from specified file.
    """
    assert (type(path) == str) and (Path(path).exists()), 'Invalid data file specified.'
    assert (type(delimiter) == str) and (0 < len(delimiter)), 'Invalid delimiter specified.'
    df = pd.read_csv(path, delimiter=delimiter)
    if ignore_header:
        df.columns = [f"x{i}" for i in range(len(df.columns) - 1)] + ["y"]
    return df


def get_dataset_mnist(batch_size: int, horizontal_flip_p: Union[int, float] = 0,
                      vertical_flip_p: Union[int, float] = 0, invert: bool = False,
                      train_store_path: str = "resources", test_store_path: str = "resources"
                      ) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST data sets (training and test).

    :param batch_size: Size of a mini-batch used by the data loaders.
    :param horizontal_flip_p: Probability of flipping images horizontally.
    :param vertical_flip_p: Probability of flipping images vertically.
    :param invert: Whether to invert the pixels of an image.
    :param train_store_path: Path where the MNIST training data will be stored.
    :param test_store_path: Path where the MNIST test data will be stored.
    :return: Tuple comprising a data loader for training [0] as well as test set [1].
    """
    assert batch_size >= 1, 'Batch size needs to be >= 1.'
    assert 0 <= horizontal_flip_p <= 1, 'Horizontal flip probability needs to be in the range [0, 1].'
    assert 0 <= vertical_flip_p <= 1, 'Vertical flip probability needs to be in the range [0, 1].'

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        lambda x: 255.0 - torch.as_tensor(x) if invert else x,
        torchvision.transforms.Normalize(mean=0.1307, std=0.3081),
        torchvision.transforms.RandomHorizontalFlip(p=horizontal_flip_p),
        torchvision.transforms.RandomVerticalFlip(p=vertical_flip_p)
    ])

    return (DataLoader(
        dataset=torchvision.datasets.MNIST(root=train_store_path, train=True, download=True, transform=transforms),
        batch_size=batch_size, shuffle=True
    ), DataLoader(
        dataset=torchvision.datasets.MNIST(root=test_store_path, train=False, download=True, transform=transforms),
        batch_size=batch_size, shuffle=False
    ))


def plot_model(dataset: pd.DataFrame, coefficients: Sequence[Sequence[Union[int, float]]],
               transform: Callable[[np.ndarray], np.ndarray] = None, transform_name: str = None,
               decimals: int = 5) -> None:
    """
    Plot (x, y) data pairs.

    :param dataset: Dataset/data frame consisting of (x, y) pairs.
    :param coefficients: Coefficients of underlying polynomial model. In case there are multiple
        models, simply pass their coefficients as a list of coefficients, i.e., each element in
        the passed argument will represent the coefficients of the corresponding model.
    :param transform: Transformation applied to the result of the model/models.
    :param transform_name: The name of the transformation function to display in the plot.
    :param decimals: Maximum number of the coefficient decimal places to display.
    """
    assert all((type(dataset) == pd.DataFrame, len(dataset.shape) == 2)), 'Invalid dataset (must be 2D pd.DataFrame).'
    model_coefficients = [coefficients] if not isinstance(coefficients[0], Sequence) else coefficients

    # Plot data pairs themselves.
    hue = dataset.columns[2] if dataset.shape[1] >= 3 else None
    _ = sns.scatterplot(data=dataset, x=dataset.columns[0], y=dataset.columns[1], hue=hue, legend=None)

    # Compute polynomial.
    y, labels = [], []
    x = np.linspace(dataset[dataset.columns[0]].min(), dataset[dataset.columns[0]].max(), 100)
    
    for i, model in enumerate(model_coefficients):
        result = np.zeros_like(x)
        label = ''
        
        for degree, coefficient in enumerate(model):
            result += coefficient * x ** degree
            sign = '+' if np.sign(coefficient) >= 0 else '-'
            if degree == 0 and sign == '+':  # special case for first coefficient (if positive, do not display +)
                sign = ''
            formatted_c = f"{round(np.abs(coefficient), decimals):.{decimals}f}".rstrip("0")
            if formatted_c.endswith("."):
                formatted_c += "0"
            label += f'{sign}{formatted_c} x^{{{degree}}}'

        prefix = '$y = ' if len(model_coefficients) == 1 else f'$y_{{{i}}} = '
        if transform_name:
            prefix += transform_name + '('
        label = prefix + label
        if transform_name:
            label += ')' if len(model_coefficients) == 1 else f')_{{{i}}}'
        label += '$'
        
        y.append(result)
        labels.append(label)

    # Plot resulting polynomial.
    if transform is not None:
        y = transform(y, axis=0)
    for current_y, current_label in zip(y, labels):
        ax = sns.lineplot(x=x, y=current_y, markers=r'-', label=current_label)
        y1, y2 = ax.get_ylim()
        ax.set_ylim(max(-10, y1), min(10, y2))


def plot_logistic_model(dataset: pd.DataFrame, coefficients: Sequence[Sequence[Union[int, float]]]) -> None:
    """
    Plot (x, y) data pairs. The output y of the underlying polynomial model will
    automatically be transformed by a softmax function before the plotting call.

    :param dataset: Dataset/data frame consisting of (x, y) pairs.
    :param coefficients: Coefficients of underlying polynomial model. In case there are multiple
        models, simply pass their coefficients as a list of coefficients, i.e., each element in
        the passed argument will represent the coefficients of the corresponding model.
    """
    plot_model(dataset, coefficients, softmax, '\u03c3')  # unicode for sigma character


def minimize_mse(dataset: pd.DataFrame, degree: int = 1) -> Tuple[Union[float, int], ...]:
    """
    Minimize the mean squared error of a dataset given a polynomial of a specified degree.

    :param dataset: Dataset/data frame consisting of (x, y) pairs.
    :param degree: Degree of the polynomial (i.e. number of coefficients - 1).
    :return: Coefficients minimizing the mean squared error of a polynomial of the specified degree.
    """
    assert all((type(dataset) == pd.DataFrame, len(dataset.shape) == 2)), 'Invalid dataset (must be 2D pd.DataFrame).'
    assert degree >= 0, 'The degree of the polynomial has to be non-negative.'
    return np.polyfit(x=dataset[dataset.columns[0]], y=dataset[dataset.columns[1]], deg=degree).tolist()[::-1]


def minimize_ce(dataset: Union[pd.DataFrame, DataLoader], iterations: int,
                learning_rate: Union[float, int], momentum: Union[float, int],
                use_cuda_if_available: bool = True) -> Tuple[Union[float, int], ...]:
    """
    Minimize the cross-entropy loss of a dataset.

    :param dataset: Dataset/data frame consisting of (x, y) pairs.
    :param iterations: Amount of iterations to minimize the cross-entropy error.
    :param learning_rate: Learning rate applied in cross-entropy minimization.
    :param momentum: Momentum term applied in cross-entropy minimization.
    :param use_cuda_if_available: Use CUDA-capable device with index 0 if available.
    :return: Coefficients minimizing the cross-entropy loss.
    """
    assert type(dataset) in (pd.DataFrame, DataLoader), 'Invalid dataset (must be pd.DataFrame or PyTorch DataLoader).'
    assert iterations >= 0, 'Iterations must be non-negative.'
    assert (type(learning_rate) in (int, float)) and learning_rate > 0, 'Learning-rate must be > 0.'
    assert (type(momentum) in (int, float)) and momentum >= 0, 'Momentum must be non-negative.'
    device = torch.device('cuda:0' if (torch.cuda.is_available() and use_cuda_if_available) else 'cpu')

    # Parse and pre-process dataset.
    if type(dataset) == pd.DataFrame:
        data = torch.from_numpy(dataset[dataset.columns[:-1]].to_numpy()).to(dtype=torch.float32)
        targets = torch.from_numpy(dataset[dataset.columns[-1]].to_numpy()).to(dtype=torch.long)
        dataset, input_size, output_size = ((data, targets),), data.shape[1], len(targets.unique())
    else:
        input_size, output_size = prod(dataset.dataset[0][0].shape), len(set(_[1] for _ in dataset.dataset))

    # Create network and auxiliary modules.
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=input_size, out_features=output_size),
        torch.nn.Softmax(dim=1)
    ).to(device=device)
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum)

    # Execute minimization step.
    for iteration in range(iterations):
        for batch_samples, batch_targets in dataset:
            batch_samples = batch_samples.to(device)
            batch_targets = batch_targets.to(device)
            loss = loss_criterion(input=model(batch_samples).squeeze(dim=1), target=batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Gather and return coefficients obtained through minimizing cross-entropy error.
    return tuple((float(_[0]), *_[1].tolist()) for _ in zip(
        model[1].bias.detach().cpu().numpy(), model[1].weight.detach().cpu().numpy()))


def predict_logistic(dataset: Union[np.ndarray, pd.DataFrame, torch.Tensor],
                     coefficients: Sequence[Sequence[Union[int, float]]]):
    """
    Get predictions for the passed dataset using a logistic model specified by the given
    coefficients. The coefficients are assumed to be a list of coefficients of the
    underlying polynomial models (one model for each possible target class). The output
    vectors y of the underlying polynomial models will automatically be transformed by a
    softmax function. Afterwards, the model with the highest score (= the highest probability)
    is chosen and its index (= the target class) is returned.
    
    :param dataset: Unlabeled dataset/data frame containing the feature vectors x.
    :param coefficients: A list containing the coefficients of each underlying polynomial model.
    :return: The prediction vector.
    """
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.values
    elif isinstance(dataset, torch.Tensor):
        dataset = dataset.numpy()
    
    # Converting a PyTorch tensor to a numpy array is actually a bit of a (slow) hack so
    # we can execute the same code below regardless. PyTorch-specific solution without
    # this initial dataset conversion would be (we still convert the final result):
    #
    # predictions = np.argmax(torch.nn.functional.softmax(torch.stack([
    #     (bias + (dataset * torch.as_tensor(weights)).sum(axis=1)) for bias, *weights in coefficients
    # ], dim=1), dim=1).numpy(), axis=1)
    
    linear_class_results = dict()
    for class_, (bias, *weights) in enumerate(coefficients):
        linear_class_results[class_] = bias + (dataset * weights).sum(axis=1)
    
    class_results_df = pd.DataFrame(linear_class_results)  # classes = columns, samples = rows
    class_results_df[:] = softmax(class_results_df, axis=1)  # = probabilities
    return class_results_df.idxmax(axis=1).values
