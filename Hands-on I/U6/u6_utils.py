# -*- coding: utf-8 -*-
"""
Authors: Brandstetter, Schäfl, Schlüter, Rumetshofer, Schörgenhumer
Date: 09-01-2023

This file is part of the "Hands on AI I" lecture material. The following copyright statement applies
to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for personal and non-commercial
educational use only. Any reproduction of this manuscript, no matter whether as a whole or in parts, no matter whether
in printed or in electronic form, requires explicit prior acceptance of the authors.
"""
import sys
from distutils.version import LooseVersion
from IPython.core.display import HTML
from typing import Callable, Tuple, Union, Dict

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.model_selection
import torch
import torchvision
import tqdm as tqdm_
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

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
    numpy_check = '(\u2713)' if LooseVersion(np.__version__) >= LooseVersion(r'1.18') else '(\u2717)'
    pandas_check = '(\u2713)' if LooseVersion(pd.__version__) >= LooseVersion(r'1.0') else '(\u2717)'
    sklearn_check = '(\u2713)' if LooseVersion(sklearn.__version__) >= LooseVersion(r'0.23') else '(\u2717)'
    matplotlib_check = '(\u2713)' if LooseVersion(matplotlib.__version__) >= LooseVersion(r'3.2.0') else '(\u2717)'
    seaborn_check = '(\u2713)' if LooseVersion(sns.__version__) >= LooseVersion(r'0.10.0') else '(\u2717)'
    torch_check = '(\u2713)' if LooseVersion(torch.__version__) >= LooseVersion(r'1.6.0') else '(\u2717)'
    tqdm_check = '(\u2713)' if LooseVersion(tqdm_.__version__) >= LooseVersion(r'4.46.0') else '(\u2717)'
    cv2_check = '(\u2713)' if LooseVersion(cv2.__version__) >= LooseVersion(r'3.4.2') else '(\u2717)'
    print(f'Installed Python version: {sys.version_info.major}.{sys.version_info.minor} {python_check}')
    print(f'Installed numpy version: {np.__version__} {numpy_check}')
    print(f'Installed pandas version: {pd.__version__} {pandas_check}')
    print(f'Installed scikit-learn version: {sklearn.__version__} {sklearn_check}')
    print(f'Installed matplotlib version: {matplotlib.__version__} {matplotlib_check}')
    print(f'Installed seaborn version: {sns.__version__} {seaborn_check}')
    print(f'Installed torch version: {torch.__version__} {torch_check}')
    print(f'Installed tqdm version: {tqdm_.__version__} {tqdm_check}')
    print(f'Installed cv2 version: {cv2.__version__} {cv2_check}')


def set_seed(seed: int = 42) -> None:
    """
    Set seed for all underlying (pseudo) random number sources.

    :param seed: seed to be used
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def show_image(image: Union[str, np.ndarray], cmap=None, figsize=(16, 8)):
    """
    Plot the specified image.
    
    :param image: Path to image file or image object (np.ndarray) to plot. If the
        input is an array, it must have either the shape (M, N) for grayscale images
        or (M, N, 3) for RGB images.
    :param cmap: Set to "gray" to convert image to grayscale. If the input is an
        array and has shape (M, N), it is automatically assumed to be a grayscale
        image and this parameter is ignored.
    :param figsize: The size of the figure in inches.
    """
    assert isinstance(image, np.ndarray) or isinstance(image, str)
    assert cmap is None or cmap == "gray"
    
    # load image
    if isinstance(image, str):
        if cmap == "gray":
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        else:
            # cv2.imread returns BGR color order so reverse it to get RGB
            image = cv2.imread(image)[:, :, ::-1]
    # infer grayscale automatically based on image shape
    elif image.ndim == 2:
        cmap = "gray"
    elif image.ndim == 3 and cmap == "gray":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # plot image
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap=cmap, vmin=0, vmax=255)
    ax.grid(None)
    ax.set_xticks(np.linspace(1, image.shape[1], 10, dtype=int))
    ax.set_yticks(np.linspace(1, image.shape[0], 10, dtype=int))
                    

def visualize_filters(filters, ncols: int = 2, image_size: float = 3, cmap="viridis", cmap_dark_to_bright=True):
    """
    Visualize filters with the numeric values of each filter component.
    
    :param filters: Sequence of filters.
    :param ncols: Number of columns of the plot (number of rows is determined automatically).
    :param image_size: Size in inches of the individual filter images.
    :param cmap: matplotlib colormap to use.
    :param cmap_dark_to_bright: Flag that indicates whether the specified colormap cmap yields
        colors from dark to bright, i.e., increasingly brighter colors. If so, the filter value
        numbers in the plot will be displayed with a white font for the lower half of the value
        range, i.e., filters.min() up to filters.min() + (filters.max() - filters.min()) / 2, and
        with a black font for the upper half of the value range. The other way (bright to dark cmap)
        is simply the inverse way (black font for lower half, white font for upper half).
    """
    n_filters = len(filters)
    assert n_filters > 0, "at least one filter must be provided"
    ncols = min(n_filters, ncols)
    nrows = int(np.ceil(n_filters / ncols))
    figsize = (ncols * image_size, nrows * image_size)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    ax = axes.flatten()

    # global vmin and vmax
    vmin = np.asarray(filters).min()
    vmax = np.asarray(filters).max()
    font_color_threshold = vmin + (vmax - vmin) / 2

    for i in range(n_filters):
        ax[i].imshow(filters[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        # axes.edgecolor is white per seaborn.set-default
        for spine in ax[i].spines.values():
            spine.set_edgecolor("black")
        ax[i].set_title(f"Filter {i + 1}")
        width, height = filters[i].shape
        for x in range(width):
            for y in range(height):
                use_white = filters[i][x][y] < font_color_threshold
                if not cmap_dark_to_bright:
                    use_white = not use_white
                ax[i].annotate(f"{filters[i][x][y]:.2f}", xy=(y, x), ha="center", va="center",
                               color="white" if use_white else "black")
    for i in range(i + 1, ax.size):
        ax[i].set_visible(False)

    
class InitializedNet(torch.nn.Module):
    """
    PyTorch network that must be initialized with supplied weights.
    """
    
    def __init__(self, weights, filter_stride=1, activation=torch.nn.ReLU(), max_pool_size=1):
        super().__init__()
        # initializes the weights of the convolutional layer to be the weights of the defined filters
        # weights must be a tensor in the correct shape (N(batch_size), C(channels), H(height), W(width))
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).float()
        if weights.ndim != 4:
            weights = weights.unsqueeze(1)
        assert weights.ndim == 4, "expected weights.shape to be either (N, H, W) or (N, C, H, W)"
        k_n, k_c, k_h, k_w = weights.shape
        self.conv = torch.nn.Conv2d(k_c, k_n, kernel_size=(k_h, k_w), stride=filter_stride, bias=False)
        self.conv.weight = torch.nn.Parameter(weights)
        self.pool = torch.nn.MaxPool2d(kernel_size=max_pool_size)
        self.activation = activation

    def forward(self, x):
        conv_x = self.conv(x)
        activated_x = self.activation(conv_x)
        pooled_x = self.pool(activated_x)
        return conv_x, activated_x, pooled_x


def visualize_cnn_layer(layer_output, n_filters: int = 20, cmap="gray", clip: bool = False, shift: int = 0,
                        ncols: int = 2, image_size: float = 7, title: str = None):
    """
    Visualize the activations of a CNN layer.
    
    :param layer_output: Tensor with the activations of a CNN layer.
    :param n_filters: Maximum number of activation maps to plot (set to None to plot all).
    :param cmap: matplotlib colormap to use.
    :param clip: If True, the output values are clipped to [0, 255], i.e., values < 0 are
        set to 0 and values > 255 are set to 255.
    :param shift: If non-zero and ``clip`` is True, an additional shift is applied to the
        output values before they are clipped to [0, 255]. This can be useful to move
        zero-centered output values to be centered, e.g., around medium gray for visualization
        purposes (shift=128), which will lead to more pronounced details being visible.
    :param ncols: Number of columns of the plot (number of rows is determined automatically).
    :param image_size: Size in inches of the individual activation map images.
    :param title: Figure title.
    """
    n_activation_maps = layer_output.shape[1]
    assert n_activation_maps > 0, "'layer' must at least contain one activation map output"
    n_filters = n_activation_maps if n_filters is None else min(n_activation_maps, n_filters)
    ncols = min(n_filters, ncols)
    nrows = int(np.ceil(n_filters / ncols))
    x_size = layer_output.shape[3]
    y_size = layer_output.shape[2]
    ratio = x_size / y_size
    figsize = (ncols * image_size, nrows * (image_size / ratio))

    # global vmin and vmax
    filters = layer_output[0, :].cpu().data.numpy()
    vmin = 0 if clip else np.asarray(filters).min()
    vmax = 255 if clip else np.asarray(filters).max()
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle(title)
    ax = axes.flatten()
    
    for i in range(n_filters):
        # grab layer outputs
        data = np.array(filters[i], copy=True)
        if clip:
            if shift != 0:
                ax[i].set_title(f"Output {i + 1}, clipped from [{data.min():.2f}, {data.max():.2f}] {shift:+} "
                                f"(shift) to [0, 255]")
                data += shift
            else:
                ax[i].set_title(f"Output {i + 1}, clipped from [{data.min():.2f}, {data.max():.2f}] to [0, 255]")
            # imshow vmin and vmax do not clip the data, they merely set the colorbar limits,
            # and the original value ranges are still kept and then scaled to vmin, vmax, so
            # we need to manually clip them here to achieve identical plots when the data
            # values are outside the vmin, vmax range
            data[data < 0] = 0
            data[data > 255] = 255
        else:
            ax[i].set_title(f"Output {i+1}, raw [{data.min():.2f}, {data.max():.2f}]")
        im = ax[i].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[i].grid(None)
        ax[i].set_xticks(np.linspace(1, layer_output.shape[3], 10, dtype=int))
        ax[i].set_yticks(np.linspace(1, layer_output.shape[2], 10, dtype=int))
    for i in range(i + 1, ax.size):
        ax[i].set_visible(False)
    
    fig.tight_layout()

    # axes.edgecolor is white per seaborn.set-default
    with plt.rc_context({"axes.edgecolor": "black"}):
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.13, 0.03, 0.74])
        cbar_ax.grid(False)  # avoid matplotlib 3.5 deprecation warning
        fig.colorbar(im, cax=cbar_ax)


def get_grayscale_image_tensor(image: Union[str, np.ndarray]):
    """
    Convert the specified image to a PyTorch tensor with a shape that can be used as
    input for a CNN model. The image will be automatically transformed into grayscale.
    
    :param image: Path to image file or image object (np.ndarray) to plot. If the
        input is an array, it must have either the shape (M, N) for grayscale images
        or (M, N, 3) for RGB images.
    :return: PyTorch tensor containing the grayscale image data.
    """
    assert isinstance(image, np.ndarray) or isinstance(image, str)
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    elif image.ndim != 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return torch.from_numpy(image).unsqueeze(0).unsqueeze(1).float()


def visualize_cnn_filters(input_, n_filters: int = 100, ncols: int = 4, image_size: float = 3, cmap="viridis"):
    """
    Visualize filters learned by a CNN.

    :param input_: Either a tensor with the weights of a CNN layer, or a CNN model layer from
        which the weights will be automatically extracted.
    :param n_filters: Maximum number of activation maps to plot (set to None to plot all).
    :param ncols: Number of columns of the plot (number of rows is determined automatically).
    :param image_size: Size in inches of the individual activation map images.
    :param cmap: matplotlib colormap to use (only used for 1D input).
    """
    if isinstance(input_, torch.nn.Module):
        layer_weights = input_.weight.detach().cpu()
    elif isinstance(input_, torch.nn.Parameter):
        layer_weights = input_.detach().cpu()
    elif isinstance(input_, np.ndarray):
        layer_weights = torch.from_numpy(input_)
    elif isinstance(input_, torch.Tensor):
        layer_weights = input_.cpu()
    else:
        raise AssertionError("input must be one of: torch.nn.Module, torch.nn.Parameter, torch.Tensor, np.ndarray")

    n_out, n_in, y_size, x_size = layer_weights.shape
    assert n_out > 0, "'layer' must at least contain one filter"
    assert n_in == 1 or n_in == 3, "can only visualize 1D or 3D input"
    n_filters = n_out if n_filters is None else min(n_out, n_filters)
    ncols = min(n_filters, ncols)
    nrows = int(np.ceil(n_filters / ncols))
    ratio = x_size / y_size
    figsize = (ncols * image_size, nrows * (image_size / ratio))

    # for 1D input, get the global vmin and vmax (for imshow, so we can use a global colorbar afterwards)
    if n_in == 1:
        vmin = layer_weights.min().item()
        vmax = layer_weights.max().item()

    # axes.edgecolor is white per seaborn.set-default
    with plt.rc_context({"axes.edgecolor": "black"}):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        ax = axes.flatten()

        for i in range(n_filters):
            if n_in == 1:
                im = ax[i].imshow(layer_weights[i, 0, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
            elif n_in == 3:
                # for RGB images, we need to fulfill the requirements by ax.imshow:
                # (M, N, 3): an image with RGB values (0-1 float or 0-255 int)
                # first reshape:       C, H, W          H, W, C
                arr = layer_weights[i, :, :, :].permute(1, 2, 0)
                # then normalize to [0, 1]
                arr = (arr - arr.min()) / (arr.max() - arr.min())
                im = ax[i].imshow(arr)
            else:
                raise AssertionError("not 1D or 3D (should have already been caught above)")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(f"Filter {i + 1}")
        for i in range(i + 1, ax.size):
            ax[i].set_visible(False)

        # only show the colorbar for 1D input
        if n_in == 1:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.13, 0.03, 0.74])
            cbar_ax.grid(False)  # avoid matplotlib 3.5 deprecation warning
            fig.colorbar(im, cax=cbar_ax)

    
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


def get_dataset(batch_size: int = 20, horizontal_flip_p: Union[int, float] = 0,
                vertical_flip_p: Union[int, float] = 0, invert: bool = False, valid_size: float = 0,
                augment_train: bool = True, augment_test: bool = False, random_state: int = 42,
                root: str = "resources", variant: str = "MNIST") -> Tuple[DataLoader, ...]:
    """
    Load data sets (training, optional validation, and test).

    :param batch_size: Size of a mini-batch used by the data loaders.
    :param horizontal_flip_p: Probability of flipping images horizontally.
    :param vertical_flip_p: Probability of flipping images vertically.
    :param augment_train: Whether to apply flipping to training images
    :param augment_test: Whether to apply flipping to test images
    :param invert: Whether to invert the pixels of an image.
    :param valid_size: Fraction of training set to keep for validation.
    :param random_state: Random state for splitting off the validation set.
    :param root: Path where the data will be stored.
    :param variant: Either "MNIST", "FashionMNIST" or "CIFAR10".
    :return: If valid_size is 0, a tuple comprising a data loader for training [0] as well as
        test set [1]. If valid_size is > 0, a tuple comprising a data loader for training [0],
        validation [1] as well as test set [2].
    """
    assert batch_size >= 1, 'Batch size needs to be >= 1.'
    assert 0 <= horizontal_flip_p <= 1, 'Horizontal flip probability needs to be in the range [0, 1].'
    assert 0 <= vertical_flip_p <= 1, 'Vertical flip probability needs to be in the range [0, 1].'
    assert 0 <= valid_size < 1, 'Validation set fraction must be in the range [0, 1)'
    assert variant in ('MNIST', 'FashionMNIST', 'CIFAR10'), \
        'Variant must be either "MNIST", "FashionMNIST" or "CIFAR10".'

    def prepare_dataset(dataset: Union[torchvision.datasets.MNIST, torchvision.datasets.FashionMNIST,
                                       torchvision.datasets.CIFAR10],
                        invert: bool, mean: float = 0.1307, std: float = 0.3081,
                        augmentations=None):
        """Takes an MNIST dataset and returns a TensorDataset with preconverted images."""
        X, Y = dataset.data, dataset.targets
        is_cifar = False
        if isinstance(X, np.ndarray):
            X = torch.Tensor(X)
            Y = torch.Tensor(Y).long()
            is_cifar = True
        if invert:
            X = X ^ 255
        X = X.float().div_(255).sub_(mean).div_(std)  # normalize
        if is_cifar:
            X = X.permute(0, 3, 1, 2)  # permute
        else:
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
                         training_set: DataLoader, iterations: Union[float, int],
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
    assert type(training_set) == DataLoader, 'Invalid dataset (must be PyTorch DataLoader).'
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
