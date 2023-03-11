# -*- coding: utf-8 -*-
"""
Authors: Brandstetter, Schäfl, Schlüter, Rumetshofer, Schörgenhumer
Date: 23-01-2023

This file is part of the "Hands on AI I" lecture material. The following copyright statement applies
to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for personal and non-commercial
educational use only. Any reproduction of this manuscript, no matter whether as a whole or in parts, no matter whether
in printed or in electronic form, requires explicit prior acceptance of the authors.
"""
import base64
import os
import random
import sys
import urllib.request
import warnings
from collections import OrderedDict
from distutils.version import LooseVersion
from IPython.core.display import HTML
from pathlib import Path
from typing import Callable, Union, Dict
from urllib.error import URLError

import fastai
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import torchvision.models as vision_models
import tqdm as tqdm_
# Import from ".all" to automatically import intermediate modules (otherwise, i.e., importing
# everything from their respective modules, it somehow does not work properly)
from fastai.vision.all import Learner, vision_learner, verify_image, error_rate, ClassificationInterpretation,\
    imagenet_stats, ImageDataLoaders, Resize, RandomResizedCrop, aug_transforms, Normalize
from fastai.data.load import DataLoader as FastaiDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm.autonotebook import tqdm
import ssl
import socket

# https://stackoverflow.com/a/59548973/8176827
socket.setdefaulttimeout(5)

# https://stackoverflow.com/a/69692664/8176827
ssl._create_default_https_context = ssl._create_unverified_context

# Must use torch.tensor(x) to create an actual torch.Tensor object, so ignore
# the corresponding warning to use .detach().clone() instead.
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r"To copy construct from a tensor, it is recommended to use sourceTensor\.clone\(\)\."
                                r"detach\(\) or sourceTensor\.clone\(\)\.detach\(\)\.requires_grad_\(True\)"
                                r", rather than torch\.tensor\(sourceTensor\)\.")
# Ignore deprecation warnings that arise internally in fastai (we have no way of fixing them anyway)
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r"The parameter 'pretrained' is deprecated since 0\.13 and will be removed in 0\.15, "
                                r"please use 'weights' instead\.")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r"Arguments other than a weight enum or `None` for 'weights' are deprecated since 0\.13"
                                r" and will be removed in 0\.15\. The current behavior is equivalent to passing "
                                r"`weights=[\w\.]+`\. You can also use `weights=[\w\.]+` to get the most up-to-date "
                                r"weights\.")


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


def check_module_versions(create_requirements_file: bool = False) -> None:
    """
    Check Python version as well as versions of recommended (partly required) modules.
    
    :param create_requirements_file: Whether to automatically create a "requirements.txt"
        file in the current working directory based on the versions of the installed packages.
        The file is only created if the installed package versions fulfill the hard-coded,
        internal requirements. If not, a warning is issued.
    """
    def check(name, version, min_version):
        if LooseVersion(version) >= LooseVersion(min_version):
            result = "\u2713"  # checkmark
        else:
            result = f"\u2717: expected at least {min_version}"  # cross
        print(f"Installed {name} version: {version} ({result})")
        return result == "\u2713"
    
    # Python check
    check("Python", sys.version.split()[0], "3.9")
    # packages check (with optionally storing the versions in the requirements file)
    checks = [(np, "1.18"),
              (pd, "1.0"),
              (mpl, "3.2.0"),
              (sns, "0.10.0"),
              (torch, "1.12.0"),
              (torchvision, "0.13.0"),
              (tqdm_, "4.46.0"),
              (fastai, "2.7.9")]
    all_fulfilled = True
    for pkg, min_v in checks:
        assert hasattr(pkg, "__name__"), f"package '{pkg}' does not have an attribute '__name__'"
        assert hasattr(pkg, "__version__"), f"package '{pkg}' does not have an attribute '__version__'"
        all_fulfilled &= check(pkg.__name__, pkg.__version__, min_v)
    if create_requirements_file:
        if all_fulfilled:
            with open("requirements.txt", "w") as f:
                f.writelines([f"{pkg.__name__}=={pkg.__version__}\n" for pkg, _ in checks])
        else:
            warnings.warn("not all version requirements fulfilled; 'requirements.txt' not created")


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


def download_all_images(path: Path, overwrite: bool = False, try_failed: bool = False,
                        failed_file_name: str = "failed.txt") -> None:
    """
    Downloads and verifies images from the URLs listed in .csv files in the given `path`.
    Failed downloads are stored in a file "path/<classname>_failed_file" and will not be
    tried to download again if this method is called multiple times unless specified otherwise.
    <classname> refers to the class indicated by the corresponding .csv file of `path`.

    :param path: The Path containing the .csv files.
    :param overwrite: If True, overwrite the image files even if they already exist.
    :param try_failed: If True, previously failed downloads are tried again.
    :param failed_file_name: The suffix of the file where failed downloads are stored.
    """
    for csv in path.glob("*.csv"):
        with open(csv) as f:
            lines = f.readlines()

        classname = csv.stem
        output_dir = path / classname
        output_dir.mkdir(exist_ok=True)

        failed_lines_file = path / f"{classname}_{failed_file_name}"
        if not failed_lines_file.exists() or try_failed:
            failed_lines = set()
        else:
            with open(failed_lines_file) as f:
                failed_lines = {line[:-1] if line.endswith("\n") else line for line in f.readlines()}
        new_failed_lines = set()
        n_ignored_failed = 0
        n_ignored_exists = 0
        
        for i, line in enumerate(tqdm(lines, f"Downloading '{classname}' images")):
            if line.endswith("\n"):
                line = line[:-1]
            file = Path()
            
            if line in failed_lines:
                n_ignored_failed += 1
            elif line.startswith("http"):
                file = output_dir / f"{i:07}.jpeg"
                if not file.exists() or overwrite:
                    try:
                        urllib.request.urlretrieve(line, file)
                        if not verify_image(file):
                            warnings.warn(f"{line}: not an image (skipping)")
                            os.remove(file)
                    except (URLError, socket.timeout) as ex:
                        warnings.warn(f"{line}: unable to access image: {ex} (skipping)")
                else:
                    n_ignored_exists += 1
            elif line.startswith("data:image/"):
                # expected line content (example): "data:image/jpeg;base64,<BASE64DATA>"
                image_format, encoding_and_data = line[11:].split(";")
                encoding, data = encoding_and_data.split(",")
                if encoding == "base64":
                    file = output_dir / f"{i:07}.{image_format}"
                    if not file.exists() or overwrite:
                        with open(file, "wb") as f:
                            f.write(base64.b64decode(data))
                        if not verify_image(file):
                            warnings.warn(f"{line}: not an image (skipping)")
                            os.remove(file)
                    else:
                        n_ignored_exists += 1
                else:
                    warnings.warn(f"{line}: unexpected encoding: {encoding} (skipping)")
            else:
                warnings.warn(f"{line}: unexpected line format (skipping)")
            
            # check if the file is there; if not, it must have failed
            if (file is None or not file.exists()) and line not in failed_lines:
                new_failed_lines.add(line)
    
        if n_ignored_failed > 0:
            print(f"ignored {n_ignored_failed} '{classname}' images due to previous download failure")
        if n_ignored_exists > 0:
            print(f"ignored {n_ignored_exists} '{classname}' images because they already exist")
        if new_failed_lines:
            print(f"{len(new_failed_lines)} '{classname}' images will be added to the set of failed images")
            
            with open(failed_lines_file, "a") as f:
                f.writelines([f"{line}\n" for line in new_failed_lines])


def load_image_dataset(path: Path, size: int = 224, batch_size: int = 32, valid_size: float = 0.2,
                       augment: bool = False, num_workers: int = 0,
                       use_cuda_if_available: bool = True) -> ImageDataLoaders:
    """
    Create image data loaders from labeled images found in `path`.

    :param path: The Path containing the image subdirectories.
    :param size: The size to resize the images to.
    :param batch_size: The number of samples of each batch.
    :param valid_size: The percentage of samples to use for validation.
    :param augment: Whether to perform image data augmentations.
    :param num_workers: Set to a positive number to use multiprocessing.
    :param use_cuda_if_available: Use CUDA-capable device with index 0 if available.
    :return: A fastai ImageDataLoaders instance.
    """
    if not augment:
        item_tfms = Resize(size, method='squish')
        batch_tfms = None
    else:
        item_tfms = RandomResizedCrop(size, min_scale=0.5)
        # see https://docs.fast.ai/vision.augment.html#aug_transforms for more
        batch_tfms = aug_transforms()
    dls = ImageDataLoaders.from_folder(path, valid_pct=valid_size, bs=batch_size, num_workers=num_workers,
                                       item_tfms=item_tfms, batch_tfms=batch_tfms, drop_last=False, seed=4711)
    dls.after_batch.add(Normalize.from_stats(*imagenet_stats))
    if not use_cuda_if_available:
        dls = dls.cpu()
    
    # Creating a new iterator implementation that simply wraps the existing one
    # and replaces fastai.TensorImage and fastai.TensorCategory objects with standard
    # pytorch Tensor objects is a hacky and computationally expensive workaround, but
    # otherwise, we get an error when calling the loss function on such objects (at
    # least with pytorch version 1.10.0 and fastai version 2.5.3 and 2.5.4; also does
    # not work with pytorch version 1.12.0 and fastai version 2.7.9; see issue
    # https://github.com/fastai/fastai/issues/3552):
    #
    # TypeError: no implementation found for 'torch.nn.functional.cross_entropy' on types
    # that implement __torch_function__: [<class 'fastai.torch_core.TensorImage'>,
    # <class 'fastai.torch_core.TensorCategory'>]
    def wrapped_iter(iter_func):
        def _iter(self):
            it = iter_func(self)
            
            class WrapperIterator:
                def __iter__(self):
                    return self
                
                def __next__(self):
                    x, y = next(it)
                    # Must use torch.tensor(x) to create an actual torch.Tensor object, so
                    # ignore the corresponding warning to use .detach().clone() instead.
                    return torch.tensor(x), torch.tensor(y)
            
            return WrapperIterator()
        
        return _iter
    
    type(dls.train).__iter__ = wrapped_iter(type(dls.train).__iter__)
    
    return dls


def plot_image_dataset(path: Path, nitems: int = 8, nrows: int = 2, size: int = 128) -> None:
    """
    Plots `nitems` labeled images found in `path`, arranged in `num_rows` rows, each
    resized to `size` by `size` pixels. The label is taken to be the directory name.
    """
    dls = load_image_dataset(path, size=size, batch_size=nitems)
    dls.train.show_batch(max_n=nitems, nrows=nrows)


def perform_magic(path: Path, iterations: int = 4, size: int = 224, batch_size: int = 32, valid_size: float = 0.2,
                  augment: bool = True, use_cuda_if_available: bool = True) -> Learner:
    """
    Trains a classifier `iterations` epochs (full iterations over the training data) on the labeled images found in
    `path`, reserving a fraction of `valid_size` images for validation. For more details on the parameters, see function
    `plot_image_dataset`.
    """
    dls = load_image_dataset(path=path, size=size, batch_size=batch_size, valid_size=valid_size,
                             augment=augment, use_cuda_if_available=use_cuda_if_available)
    learner = vision_learner(dls, vision_models.resnet34, metrics=error_rate)
    if not use_cuda_if_available:
        learner = learner.cpu()
    learner.fit_one_cycle(iterations)
    return learner


def evaluate_classifier(learner: Learner) -> ClassificationInterpretation:
    """
    Returns a fastai ClassificationInterpretation for a given `learner`.
    """
    return ClassificationInterpretation.from_learner(learner)


def create_cnn(num_classes: int, num_layers: int = 5, dropout: float = 0, batchnorm: bool = False,
               residuals: bool = False, pretrained: bool = False) -> nn.Module:
    """
    Create a CNN classification model.

    :param num_classes: Number of target classes (= number of output units).
    :param num_layers: Number of layers, supports 5, 18, 34, 50, 101 or 152.
    :param dropout: How much dropout to use (try 0.5 or 0.25).
    :param batchnorm: Whether to use batch normalization.
    :param residuals: Whether to use residual connections.
    :param pretrained: Whether to use pretrained weights (and freeze them).
    :return: A PyTorch neural network model.
    """

    def remove_layers(module, cls):
        # recursively remove layers of class `cls` from module
        # first search for layers (and remove in submodules)
        remove = []
        for name, submodule in module.named_children():
            if isinstance(submodule, cls):
                remove.append(name)
            else:
                remove_layers(submodule, cls)
        # then remove if possible, replace with Identity otherwise
        if remove:
            if isinstance(module, nn.Sequential):
                for name in remove:
                    delattr(module, name)
            else:
                for name in remove:
                    setattr(module, name, nn.Identity())

    def add_biases(module):
        for m in module.modules():
            if hasattr(m, 'bias') and m.bias is None:
                m.bias = nn.Parameter(torch.zeros(m.weight.shape[0]))

    def remove_residuals(module):
        # replace BasicBlock instances with Sequentials
        for name, submodule in list(module.named_children()):
            if isinstance(submodule, vision_models.resnet.BasicBlock):
                setattr(module, name, nn.Sequential(OrderedDict([
                    (name, getattr(submodule, name))
                    for name in ('conv1', 'bn1', 'relu', 'conv2', 'bn2')] +
                    [('relu2', submodule.relu)])))
            else:
                remove_residuals(submodule)

    # create base model
    if num_layers == 5:
        # manually defined 5-layer model
        if residuals:
            raise NotImplementedError("residuals not implemented for the 5-layer model")
        if pretrained:
            raise NotImplementedError("pretrained weights not available for the 5-layer model")
        model = nn.Sequential(OrderedDict([
            ('body', nn.Sequential(
                nn.Conv2d(3, 8, 5, bias=not batchnorm),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.MaxPool2d(3),
                nn.Conv2d(8, 16, 5, bias=not batchnorm),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(3),
                nn.Conv2d(16, 32, 5, bias=not batchnorm),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(3))),
            ('head', nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(32, 64, bias=not batchnorm),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)))
        ]))
    elif hasattr(vision_models, 'resnet%d' % num_layers):
        # resnet model for ImageNet (use v1 weight version to use the same as fastai)
        model = getattr(vision_models, 'resnet%d' % num_layers)(weights="IMAGENET1K_V1" if pretrained else None)
        # replace classification layer with the correct number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # freeze weights if pretrained (except for the classification layer)
        if pretrained:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.fc.parameters():
                p.requires_grad = True
        # add dropout if needed
        if dropout:
            model.fc = nn.Sequential(nn.Dropout(dropout), model.fc)
    else:
        raise ValueError("num_layers=%d is unsupported" % num_layers)

    # remove unwanted layers again (this is easier than not adding them)
    if not dropout:
        remove_layers(model, nn.Dropout)
    if not batchnorm:
        remove_layers(model, (nn.BatchNorm2d, nn.BatchNorm1d))
        add_biases(model)
    if not residuals and num_layers != 5:
        remove_residuals(model)

    return model


def run_gradient_descent(model: torch.nn.Module, loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                         training_set: Union[TorchDataLoader, FastaiDataLoader], iterations: int,
                         learning_rate: Union[float, int], momentum: Union[float, int],
                         valid_set: Union[TorchDataLoader, FastaiDataLoader] = None, lr_schedule: str = None,
                         plot_curves: bool = False, use_cuda_if_available: bool = True,
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
    :param lr_schedule: A learning rate schedule to use: None, "linear", "steps" or "onecycle".
    :param plot_curves: If True, plot loss and learning rate curves when finished.
    :param show_batch_progress: If True, the progress bar will show the number of batches. If
        False, the progress par will show the number of individual samples.
    :return: Loss per epoch.
    """
    assert isinstance(training_set, (TorchDataLoader, FastaiDataLoader)),\
        f'Invalid dataset (must be PyTorch DataLoader or fastai DataLoader, not {type(training_set)}).'
    assert iterations >= 0, 'Iterations must be non-negative.'
    assert (type(learning_rate) in (int, float)) and learning_rate > 0, 'Learning-rate must be > 0.'
    assert (type(momentum) in (int, float)) and momentum >= 0, 'Momentum must be non-negative.'
    device = torch.device('cuda:0' if (torch.cuda.is_available() and use_cuda_if_available) else 'cpu')

    # move model to GPU if available (we already set the device accordingly)
    model = model.to(device)

    # instantiate optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum)

    # instantiate scheduler
    total_steps = iterations * len(training_set)
    if lr_schedule == "linear":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / total_steps)
        schedule_at = "batch"
    elif lr_schedule == "steps":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        schedule_at = "epoch"
    elif lr_schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)
        schedule_at = "batch"
    else:
        schedule_at = "never"
    
    # run training loop
    batch_size = training_set.batch_size if isinstance(training_set, TorchDataLoader) else training_set.bs
    pbar = tqdm(total=len(training_set) if show_batch_progress else len(training_set.dataset))
    assert int(np.ceil(len(training_set.dataset) / batch_size)) == len(training_set)
    errors = []
    valid_errors = []
    learning_rates = []
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
            learning_rates.append(optimizer.param_groups[0]['lr'])
            if schedule_at == "batch":
                scheduler.step()
        errors[-1] /= len(training_set)
        if valid_set is not None:
            valid_errors.append(evaluate_model(model, valid_set, loss=loss)['loss'])
        print(f'Epoch {epoch + 1:2d} finished with training loss: {errors[-1]:.6f}' +
              (f' and validation loss: {valid_errors[-1]:.6f}' if valid_set else ''))
        if schedule_at == "epoch":
            scheduler.step(errors[-1] if valid_set is None else valid_errors[-1])
    pbar.close()
    
    # compile training curves
    curves = {'training loss': np.asarray(errors)}
    if valid_set is not None:
        curves['validation loss'] = np.asarray(valid_errors)
    curves = pd.DataFrame(curves, index=np.arange(1, iterations + 1))
    
    # plot curves if requested
    if plot_curves:
        sns.lineplot(data=curves)
        plt.show()
    
    return curves


def evaluate_model(model: torch.nn.Module, dataset: Union[TorchDataLoader, FastaiDataLoader],
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


def multiclass_error_rate(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the multi-class error rate for a given set of samples. This is simply 1 - `multiclass_accuracy`.

    :param preds: Predictions as an NxC matrix of class probabilities.
    :param targets: Targets as an N-dimensional vector of integers.
    :return: Average error rate for the given samples.
    """
    return 1 - multiclass_accuracy(preds, targets)


def visualize_cnn_filters(model: nn.Module, index: int = 0, n_filters: int = 100,
                          ncols: int = 4, image_size: float = 3, cmap="viridis") -> torch.Tensor:
    """
    Visualize filters learned by a CNN and return the filters.

    :param model: The model to visualize a conv layer for.
    :param index: Which conv layer to visualize (0 for the first, 1 for the second, etc.).
    :param n_filters: Maximum number of activation maps to plot (set to None to plot all).
    :param ncols: Number of columns of the plot (number of rows is determined automatically).
    :param image_size: Size in inches of the individual activation map images.
    :param cmap: matplotlib colormap to use (only used for 1D input).
    :return: All filters/weights of the chosen layer.
    """
    conv_layers = [l for l in model.modules() if isinstance(l, nn.Conv2d)]
    layer = conv_layers[index]
    layer_weights = layer.weight.detach().cpu()

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
    
    return layer_weights.clone()
