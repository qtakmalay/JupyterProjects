# -*- coding: utf-8 -*-
"""
Authors: Brandstetter, Schäfl, Schlüter, Mitterecker, Ramsauer, Rumetshofer, Winter, Patil, Schörgenhumer
Date: 05-12-2022

This file is part of the "Hands on AI I" lecture material. The following copyright statement applies
to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for personal and non-commercial
educational use only. Any reproduction of this manuscript, no matter whether as a whole or in parts, no matter whether
in printed or in electronic form, requires explicit prior acceptance of the authors.
"""
import math
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.model_selection
import scipy
import numbers
import sys
import itertools
import torch
import torchvision
import tqdm as tqdm_

from distutils.version import LooseVersion
from IPython.core.display import HTML
from torch.utils.data import DataLoader
from typing import Callable, Sequence, Tuple, Union, Dict, List
from tqdm.autonotebook import tqdm


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
    tqdm_check = '(\u2713)' if LooseVersion(tqdm_.__version__) >= LooseVersion(r'4.46.0') else '(\u2717)'
    print(f'Installed Python version: {sys.version_info.major}.{sys.version_info.minor} {python_check}')
    print(f'Installed numpy version: {np.__version__} {numpy_check}')
    print(f'Installed pandas version: {pd.__version__} {pandas_check}')
    print(f'Installed scikit-learn version: {sklearn.__version__} {sklearn_check}')
    print(f'Installed matplotlib version: {matplotlib.__version__} {matplotlib_check}')
    print(f'Installed seaborn version: {sns.__version__} {seaborn_check}')
    print(f'Installed scipy version: {scipy.__version__} {scipy_check}')
    print(f'Installed torch version: {torch.__version__} {torch_check}')
    print(f'Installed tqdm version: {tqdm_.__version__} {tqdm_check}')


def set_seed(seed: int = 42) -> None:
    """
    Set seed for all underlying (pseudo) random number sources.

    :param seed: seed to be used
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


def get_dataset_logistic(num_pairs: int, threshold: float = 0.5, variance: float = 0) -> pd.DataFrame:
    """
    Create classification dataset consisting of randomly generated (x, y) pairs.

    :param num_pairs: Amount of (x, y) pairs to generate.
    :param threshold: Position of the class boundary.
    :param variance: Amount of noise applied in deriving the labels.
    :return: Dataset/data frame consisting of randomly generated (x, y) pairs.
    """
    assert num_pairs >= 1, 'At least one pair has to be generated.'
    x = np.random.rand(num_pairs)
    y = x
    if variance:
        y = y + np.random.randn(*y.shape) * variance
    y = (y > threshold).astype(x.dtype)
    return pd.DataFrame({"x": x, "y": y}, dtype=np.float32)


def plot_loss_landscape(loss_fn: Callable, dataset: pd.DataFrame, grad_fn: Callable = None,
                        cmap='magma', **kwargs) -> pd.DataFrame:
    """
    Compute and plot a loss landscape as a 2-dimensional image.
    
    :param loss_fn: The loss function (must accept a dataset and two vectorized model parameters).
    :param dataset: The dataset to pass to the loss function.
    :param grad_fn: The gradient function (optional, must accept a dataset and two vectorized model parameters).
    :param cmap: The color map to use (try "gist_earth" for a terrain map feel).
    :param kwargs: Two model parameters, each given a vector of values to try.
    :return: The loss landscape as a DataFrame.
    """
    assert len(kwargs) == 2, "There must be exactly 2 model parameter vectors specified as kwargs."
    param_names, param_values = zip(*kwargs.items())
    
    # pass to the loss function to compute the loss landscape
    # we make use of numpy's broadcasting to compute all the losses at once
    landscape = loss_fn(dataset, **{param_names[0]: param_values[0][:, np.newaxis, np.newaxis],
                                    param_names[1]: param_values[1][np.newaxis, :, np.newaxis]})
    if grad_fn is not None:
        grads = grad_fn(dataset, **{param_names[0]: param_values[0][::10, np.newaxis, np.newaxis],
                                    param_names[1]: param_values[1][np.newaxis, ::10, np.newaxis]})
    
    # show it
    w, h = plt.rcParams['figure.figsize']
    fig = plt.figure(figsize=(2 * w, h))
    # - 2D version
    extent = (param_values[1][0], param_values[1][-1],  # left, right
              param_values[0][0], param_values[0][-1])  # bottom, top
    with sns.axes_style("dark"):
        ax = fig.add_subplot(121)
    img = ax.imshow(landscape, extent=extent, cmap=cmap, origin='lower', aspect='auto')
    cont = ax.contour(landscape, colors="white", alpha=0.5, linestyles=':', extent=extent, origin='lower')
    if grad_fn is not None:
        ax.quiver(param_values[1][::10], param_values[0][::10], grads[1], grads[0], color='white')
    ax.set_ylabel(param_names[0])
    ax.set_xlabel(param_names[1])
    cbar = fig.colorbar(img, label='loss')
    cbar.add_lines(cont)
    cbar.lines[0].set_linestyles(cont.linestyles)
    # - 3D version
    with sns.axes_style("white"):
        ax = fig.add_subplot(122, projection='3d')
    Y, X = np.meshgrid(param_values[0], param_values[1])
    ax.plot_surface(X, Y, landscape.T, cmap=cmap, linewidth=0)
    ax.set_ylabel(param_names[0])
    ax.set_xlabel(param_names[1])
    ax.set_zlabel('loss', rotation=90)
    
    # and return it
    df = pd.DataFrame(data=landscape, index=param_values[0], columns=param_values[1])
    df.index.name = param_names[0]
    df.columns.name = param_names[1]
    return df


def plot_gradient_descent(loss_fn: Callable, grad_fn: Callable, dataset: pd.DataFrame,
                          steps: int = 100, stepsize: float = 0.01, momentum: float = 0.0,
                          cmap='magma', **kwargs) -> tuple:
    """
    Compute and plot the trajectory of gradient descent on a loss surface.

    :param loss_fn: The loss function (must accept a dataset and two vectorized model parameters).
    :param grad_fn: The gradient function (must accept a dataset and two vectorized model parameters).
    :param dataset: The dataset to pass to the loss function.
    :param steps: The number of gradient descent steps.
    :param stepsize: The learning rate for gradient descent.
    :param momentum: The momentum for gradient descent.
    :param cmap: The color map to use (try "gist_earth" for a terrain map feel).
    :kwargs: Initial values for the two parameters to optimize.
    :return: The final position in parameter space.
    """
    assert len(kwargs) == 2, "There must be exactly 2 model parameter values specified as kwargs."
    
    # run gradient descent
    param_names, start = zip(*kwargs.items())
    start = np.asarray(start, np.float32)
    trajectory = [start]
    position = start
    velocity = np.zeros_like(start)
    for _ in range(steps):
        grad = np.asarray(grad_fn(dataset, position[0], position[1]), np.float32)
        velocity = velocity * momentum - grad * (1 - momentum)
        position = position + stepsize * velocity
        trajectory.append(position)
    
    # compute height dimensions for trajectory
    trajectory = np.stack(trajectory)
    heights = loss_fn(dataset, trajectory[:, 0, np.newaxis], trajectory[:, 1, np.newaxis])
    
    # plot landscape around trajectory
    param_values = (np.linspace(min(-10, trajectory[:, 0].min() - 1), max(10, trajectory[:, 0].max() + 1), 101),
                    np.linspace(min(-10, trajectory[:, 1].min() - 1), max(10, trajectory[:, 1].max() + 1), 101))
    plot_loss_landscape(loss_fn, dataset, cmap=cmap, **{param_names[0]: param_values[0],
                                                        param_names[1]: param_values[1]})
    
    # plot trajectory
    ax2d, *_, ax3d = plt.gcf().axes
    ax2d.plot(trajectory[:, 1], trajectory[:, 0], '.-', color='white', alpha=0.5, markevery=5)
    ax3d.plot(trajectory[:, 1], trajectory[:, 0], heights, '.-', color='white', alpha=0.5, markevery=5)
    
    # return final position after optimization
    return tuple(position)


def plot_model(dataset: pd.DataFrame, coefficients: Sequence[Sequence[Union[int, float]]],
               transform: Callable[[np.ndarray], np.ndarray] = None, decimals: int = 5) -> None:
    """
    Plot (x, y) data pairs along with model predictions.

    :param dataset: Dataset/data frame consisting of (x, y) pairs.
    :param coefficients: Coefficients of underlying polynomial model. In case there are multiple
        models, simply pass their coefficients as a list of coefficients, i.e., each element in
        the passed argument will represent the coefficients of the corresponding model.
    :param transform: Transformation applied to the result of the model/models.
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
    
    for model in model_coefficients:
        result, label = np.zeros_like(x), '$y = '
        if transform is scipy.special.expit:
            label += r'\sigma('
        elif transform is not None:
            label += 'f('
        for degree, coefficient in enumerate(model):
            result += coefficient * x ** degree
            sign = '+' if np.sign(coefficient) >= 0 else '-'
            if degree == 0 and sign == '+':  # special case for first coefficient (if positive, do not display +)
                sign = ''
            formatted_c = f"{round(np.abs(coefficient), decimals):.{decimals}f}".rstrip("0")
            if formatted_c.endswith("."):
                formatted_c += "0"
            label += f'{sign}{formatted_c} x^{{{degree}}}'
        if transform is not None:
            label += ')'
        label += '$'
        y.append(result)
        labels.append(label)
    
    # Plot resulting polynomial.
    if transform is not None:
        y = transform(np.asarray(y))
    for current_y, current_label in zip(y, labels):
        _ = sns.lineplot(x=x, y=current_y, markers=r'-', label=current_label)


class AugmentedTensorDataset(torch.utils.data.TensorDataset):
    """
    Subclass of TensorDataset that includes an input transformation function.
    """
    
    def __init__(self, *tensors: torch.Tensor, transform_input: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__(*tensors)
        self.transform_input = transform_input
    
    def __getitem__(self, obj):
        item = super().__getitem__(obj)
        if self.transform_input:
            item = (self.transform_input(item[0]),) + item[1:]
        return item


def get_dataset_mnist(batch_size: int = 20, horizontal_flip_p: Union[int, float] = 0,
                      vertical_flip_p: Union[int, float] = 0, invert: bool = False, valid_size: float = 0,
                      augment_train:bool = True, augment_test: bool = False, random_state: int = 42,
                      root: str = "resources", variant: str = "MNIST") -> Tuple[DataLoader, ...]:
    """
    Load MNIST data sets (training, optional validation, and test).

    :param batch_size: Size of a mini-batch used by the data loaders.
    :param horizontal_flip_p: Probability of flipping images horizontally.
    :param vertical_flip_p: Probability of flipping images vertically.
    :param augment_train: Whether to apply flipping to training images
    :param augment_test: Whether to apply flipping to test images
    :param invert: Whether to invert the pixels of an image.
    :param valid_size: Fraction of training set to keep for validation.
    :param random_state: Random state for splitting off the validation set.
    :param root: Path where the data will be stored.
    :param variant: Either "MNIST" or "FashionMNIST".
    :return: If valid_size is 0, a tuple comprising a data loader for training [0] as well as
        test set [1]. If valid_size is > 0, a tuple comprising a data loader for training [0],
        validation [1] as well as test set [2].
    """
    assert batch_size >= 1, 'Batch size needs to be >= 1.'
    assert 0 <= horizontal_flip_p <= 1, 'Horizontal flip probability needs to be in the range [0, 1].'
    assert 0 <= vertical_flip_p <= 1, 'Vertical flip probability needs to be in the range [0, 1].'
    assert 0 <= valid_size < 1, 'Validation set fraction must be in the range [0, 1)'
    assert variant in ('MNIST', 'FashionMNIST'), 'Variant must be either "MNIST" or "FashionMNIST".'
    
    def prepare_dataset(dataset: Union[torchvision.datasets.MNIST, torchvision.datasets.FashionMNIST],
                        invert: bool, mean: float = 0.1307, std: float = 0.3081,
                        augmentations=None):
        """Takes an MNIST dataset and returns a TensorDataset with preconverted images."""
        X, Y = dataset.data, dataset.targets
        if invert:
            X = X ^ 255
        X = X.float().div_(255).sub_(mean).div_(std)  # normalize
        X = X[:, np.newaxis]  # insert channel dimension
        return AugmentedTensorDataset(X, Y, transform_input=augmentations)
    
    # on-the-fly transformations
    augmentations = []
    if horizontal_flip_p:
        augmentations.append(torchvision.transforms.RandomHorizontalFlip(p=horizontal_flip_p))
    if vertical_flip_p:
        augmentations.append(torchvision.transforms.RandomVerticalFlip(p=vertical_flip_p))
    if augmentations:
        augmentations = torchvision.transforms.Compose(augmentations)
    else:
        augmentations = None
    
    datasetclass = getattr(torchvision.datasets, variant)
    
    loaders = []
    trainset = datasetclass(root=root, train=True, download=True)
    trainset = prepare_dataset(trainset, invert, augmentations=augment_train and augmentations)
    if valid_size:
        train_idxs, valid_idxs = sklearn.model_selection.train_test_split(np.arange(len(trainset)),
                                                                          test_size=valid_size,
                                                                          random_state=random_state,
                                                                          stratify=trainset.tensors[1].numpy())
        validset = torch.utils.data.Subset(trainset, valid_idxs)
        trainset = torch.utils.data.Subset(trainset, train_idxs)
    loaders.append(DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True))
    if valid_size:
        loaders.append(DataLoader(dataset=validset, batch_size=batch_size, shuffle=False))
    testset = datasetclass(root=root, train=False, download=True)
    testset = prepare_dataset(testset, invert, augmentations=augment_test and augmentations)
    loaders.append(DataLoader(dataset=testset, batch_size=batch_size, shuffle=False))
    return tuple(loaders)


def run_gradient_descent(model: torch.nn.Module, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                         training_set: DataLoader, iterations: int,
                         learning_rate: Union[float, int], momentum: Union[float, int],
                         valid_set: DataLoader = None, use_cuda_if_available: bool = False,
                         show_batch_progress: bool = False) -> pd.DataFrame:
    """
    Minimize the loss of a model on a dataset.

    :param model: The model to optimize the parameters of.
    :param loss: The loss function to minimize.
    :param training_set: DataLoader for the training data.
    :param iterations: Amount of epochs (full iterations over the training data).
    :param learning_rate: Learning rate for the update steps.
    :param momentum: Momentum term for the update steps.
    :param valid_set: DataLoader for the validation data (optional).
    :param use_cuda_if_available: Use CUDA-capable device with index 0 if available.
    :param show_batch_progress: If True, the progress bar will show the number of batches. If
        False, the progress par will show the number of individual samples.
    :return: Loss per epoch.
    """
    assert isinstance(training_set, DataLoader), 'Invalid dataset (must be PyTorch DataLoader).'
    assert iterations >= 0, 'Iterations must be non-negative.'
    assert (type(learning_rate) in (int, float)) and learning_rate > 0, 'Learning-rate must be > 0.'
    assert (type(momentum) in (int, float)) and momentum >= 0, 'Momentum must be non-negative.'
    device = torch.device('cuda:0' if (torch.cuda.is_available() and use_cuda_if_available) else 'cpu')
    
    # move model to GPU if available (we already set the device accordingly)
    model = model.to(device)
    
    # instantiate optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum)
    
    # run training loop
    pbar = tqdm(total=len(training_set) if show_batch_progress else len(training_set.dataset))
    assert int(np.ceil(len(training_set.dataset) / training_set.batch_size)) == len(training_set)
    errors = []
    valid_errors = []
    for epoch in range(iterations):
        pbar.set_description(f'Epoch {epoch + 1}/{iterations}')
        pbar.reset()
        errors.append(0)
        model.train(True)
        for inputs, targets in training_set:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            error = loss(preds.squeeze(dim=1), targets)
            errors[-1] += error.item()
            error.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1 if show_batch_progress else len(inputs))
        errors[-1] /= len(training_set)
        if valid_set is not None:
            valid_errors.append(evaluate_model(model, valid_set, loss=loss)['loss'])
        print(f'Epoch {epoch + 1:2d} finished with training loss: {errors[-1]:.6f}' +
              (f' and validation loss: {valid_errors[-1]:.6f}' if valid_set else ''))
    pbar.close()
    
    # return training curve
    curves = {'training loss': np.asarray(errors)}
    if valid_set is not None:
        curves['validation loss'] = np.asarray(valid_errors)
    return pd.DataFrame(curves, index=np.arange(1, iterations + 1))


def evaluate_model(model: torch.nn.Module, dataset: DataLoader,
                   **losses: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Dict[str, float]:
    """
    Computes one or more loss functions for a model on a dataset.
    
    :param model: The model to optimize the parameters of.
    :param dataset: DataLoader for the evaluation data.
    :param losses: The loss functions to compute.
    :return: A float for each given loss function.
    """
    device = next(model.parameters()).device
    
    results = {name: 0.0 for name in losses}
    model.train(False)
    for inputs, targets in dataset:
        inputs = inputs.to(device)
        targets = targets.to(device)
        for name, loss in losses.items():
            with torch.no_grad():
                preds = model(inputs)
                results[name] += loss(preds.squeeze(dim=1), targets).item()
    results = {name: loss / len(dataset)
               for name, loss in results.items()}
    return results


def multiclass_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the multi-class accuracy for a given set of samples.
    
    :param preds: Predictions as an NxC matrix of class probabilities.
    :param targets: Targets as an N-dimensional vector of integers.
    :return: Average accuracy for the given samples.
    """
    preds = preds.argmax(-1)
    return (preds == targets).float().mean()


def visualize_model(sizes: List[int], diameter: float = 0.1, hdist: float = 0.15, vdist: float = 0.5, dpi=180):
    """
    Draw a fully-connected neural network model.
    
    :param sizes: The number of nodes per layer.
    :param diameter: The size of each node in inches.
    :param hdist: The horizontal space between node centers in inches.
    :param vdist: The vertical space between node centers in inches.
    :param dpi: The resolution in dots per inches.
    """
    sizes = np.asarray(sizes)[::-1]  # reverse order so the display of nodes is from top to bottom
    num_layers = len(sizes)
    # precompute the width required for each layer
    layer_widths = (sizes - 1) * hdist + diameter
    # compute the total figure size and create the figure
    figheight = (num_layers - 1) * vdist + diameter
    figwidth = max(layer_widths)
    fig = plt.figure(figsize=(figwidth, figheight), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    # precompute position of center of leftmost node per layer
    x_offsets = .5 * diameter + (figwidth - layer_widths) / 2
    y_offsets = .5 * diameter + vdist * np.arange(num_layers)
    # precompute x, y positions of all nodes
    nodes = [list(zip((x_offset + hdist * np.arange(size)) / figwidth, itertools.repeat(y_offset / figheight)))
             for x_offset, y_offset, size in zip(x_offsets, y_offsets, sizes)]
    # draw
    for layer in range(num_layers):
        if layer < len(sizes) - 1:
            # draw connections to next layer
            ax.add_collection(matplotlib.collections.LineCollection(
                itertools.product(nodes[layer], nodes[layer + 1]),
                color='black', linewidth=0.1, alpha=0.8))
        # draw nodes themselves
        ax.scatter(*zip(*nodes[layer]), s=diameter * 400, color='#0078AA', zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_input_weights(model: Union[torch.nn.Linear, torch.nn.Sequential], input_shape: Tuple[int, int],
                       max_num: int = 100, ncols: int = 5, image_size: float = 2):
    """
    Visualize the input weights of a fully-connected neural network model in PyTorch.

    :param model: The model to visualize.
    :param input_shape: Shape of the input image.
    :param max_num: Maximum number of weights/nodes to visualize.
    :param ncols: Number of columns of the plot (number of rows is determined automatically).
    :param image_size: Size in inches of the individual weight/node images.
    """
    if hasattr(model, "weight"):
        weights = model.weight
    else:
        weights = next(layer.weight for layer in model if hasattr(layer, "weight"))
    weights = weights.detach().cpu().numpy()
    
    num_nodes = min(max_num, len(weights))
    assert num_nodes >= 1, "model does not contain any weights"
    if num_nodes < len(weights):
        warnings.warn(f"only showing max_num={max_num} nodes out of the possible {len(weights)} nodes")
    
    ncols = min(num_nodes, ncols)
    nrows = math.ceil(num_nodes / ncols)
    figsize = (ncols * image_size, nrows * image_size)
    
    vmin = weights.min()
    vmax = weights.max()
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    for i in range(num_nodes):
        im = axes[i].imshow(weights[i].reshape(input_shape), cmap="RdBu", vmin=vmin, vmax=vmax)
        axes[i].axis("off")
    for i in range(i + 1, axes.size):
        axes[i].set_visible(False)
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.13, 0.03, 0.74])
    cbar_ax.grid(False)  # avoid matplotlib 3.5 deprecation warning
    fig.colorbar(im, cax=cbar_ax)
    plt.show()


def get_dataset_blob2d(num_samples: int, threshold: float, offset: Tuple[float, float] = (0, 0),
                       variance: Union[int, float] = 0) -> pd.DataFrame:
    """
    Create dataset consisting of randomly generated (x1, x2, y) samples.
    Class 1 is a circle around the given offset with given threshold (radius), other samples are of class 0.

    :param num_samples: Amount of samples to generate.
    :param threshold: Boundary radius of class 1.
    :param offset: Center of class 1.
    :param variance: Amount of noise applied in deriving the labels.
    :return: Dataset/data frame consisting of randomly generated (x1, x2, y) samples.
    """
    assert num_samples >= 1, 'At least one sample has to be generated.'
    assert variance >= 0, 'Variance has to be non-negative.'
    x1 = np.random.randn(num_samples)
    x2 = np.random.randn(num_samples)
    y = np.sqrt((x1 - offset[0]) ** 2 + (x2 - offset[1]) ** 2)
    if variance:
        y = y + np.random.randn(*y.shape) * variance
    y = (y < threshold).astype(x1.dtype)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y}, dtype=np.float32)
