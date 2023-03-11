# -*- coding: utf-8 -*-
"""
Authors: B. Schäfl, S. Lehner, J. Brandstetter, A. Schörgenhumer
Date: 07-03-2023

This file is part of the "Hands on AI II" lecture material. The following copyright statement applies
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
import PIL
import scipy
import seaborn as sns
import sklearn
import sys
import torch
import random

from distutils.version import LooseVersion
from IPython.core.display import HTML
from itertools import product
from matplotlib.colors import ListedColormap
from operator import gt, lt
from pathlib import Path
from PIL import Image, ImageFilter
from sklearn import datasets
from sklearn.base import ClassifierMixin
from sklearn.cluster import KMeans, AffinityPropagation
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


def setup_covid_dashboard() -> HTML:
    """
    Setup COVID-19 dashboard as provided by the Johns Hopkins University Center
    for Systems Science and Engineering (JHU CSSE) [1].

    [1] Dong E, Du H, Gardner L. An interactive web-based dashboard to track COVID-19 in real time.
        Lancet Inf Dis. 20(5):533-534. doi: 10.1016/S1473-3099(20)30120-1

    :return: HTML instance comprising the COVID-19 dashboard from JHU CSSE [1]
    """
    return HTML("""
    <style>
        .embed-container {position: relative; padding-bottom: 80%; height: 0; max-width: 100%;}
        .embed-container iframe,
        .embed-container object,
        .embed-container iframe{position: absolute; top: 0; left: 0; width: 100%; height: 100%;}
        small{position: absolute; z-index: 40; bottom: 0; margin-bottom: -15px;}
    </style>
    <div class="embed-container">
        <iframe width="500" height="400" frameborder="0" scrolling="no"
                marginheight="0" marginwidth="0" title="COVID-19"
                src="https://www.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6">
        </iframe>
    </div>
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
    sklearn_check = '(\u2713)' if LooseVersion(sklearn.__version__) >= LooseVersion('0.23') else '(\u2717)'
    scipy_check = '(\u2713)' if LooseVersion(scipy.__version__) >= LooseVersion('1.5.0') else '(\u2717)'
    matplotlib_check = '(\u2713)' if LooseVersion(matplotlib.__version__) >= LooseVersion('3.2.0') else '(\u2717)'
    seaborn_check = '(\u2713)' if LooseVersion(sns.__version__) >= LooseVersion('0.10.0') else '(\u2717)'
    pil_check = '(\u2713)' if LooseVersion(PIL.__version__) >= LooseVersion('6.0.0') else '(\u2717)'
    print(f'Installed Python version: {sys.version_info.major}.{sys.version_info.minor} {python_check}')
    print(f'Installed numpy version: {np.__version__} {numpy_check}')
    print(f'Installed pandas version: {pd.__version__} {pandas_check}')
    print(f'Installed PyTorch version: {torch.__version__} {torch_check}')
    print(f'Installed scikit-learn version: {sklearn.__version__} {sklearn_check}')
    print(f'Installed scipy version: {scipy.__version__} {scipy_check}')
    print(f'Installed matplotlib version: {matplotlib.__version__} {matplotlib_check}')
    print(f'Installed seaborn version: {sns.__version__} {seaborn_check}')
    print(f'Installed PIL version: {PIL.__version__} {pil_check}')


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


def load_wine() -> pd.DataFrame:
    """
    Load wine dataset [1].

    [1] Forina, M. et al, PARVUS - An Extendible Package for Data Exploration, Classification and Correlation.
        Institute of Pharmaceutical and Food Analysis and Technologies, Via Brigata Salerno, 16147 Genoa, Italy.

    :return: wine dataset
    """
    wine_data = datasets.load_wine()
    data = pd.DataFrame(wine_data['data'], columns=wine_data['feature_names'])
    data['cultivator'] = wine_data['target']
    return data


def load_iris() -> pd.DataFrame:
    """
    Load iris dataset [1].

    [1] Fisher,R.A. - The use of multiple measurements in taxonomic problems.
        Annual Eugenics, 7, Part II, 179-188 (1936)

    :return: iris dataset
    """
    iris_data = datasets.load_iris()
    new_col_names = [c.replace(" (cm)", "") for c in iris_data["feature_names"]]
    data = pd.DataFrame(iris_data["data"], columns=new_col_names)
    data["species"] = iris_data["target"]
    return data


def load_breast_cancer() -> pd.DataFrame:
    """
    Load breast cancer wisconsin (diagnostic) dataset [1].

    [1] W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction for breast tumor diagnosis.
        IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905,
        pages 861-870, San Jose, CA, 1993.

    :return: breast cancer dataset
    """
    bc_data = datasets.load_breast_cancer()
    data = pd.DataFrame(bc_data['data'], columns=bc_data['feature_names'])
    data['class'] = bc_data['target']
    return data


def load_fashion_mnist() -> pd.DataFrame:
    """
    Load Fashion-MNIST data set [1].

    [1] Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
        Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747

    :return: Fashion-MNIST data set
    """
    fashion_mnist_data = datasets.fetch_openml(name=r'Fashion-MNIST', as_frame=True)
    data = fashion_mnist_data["data"].astype(np.float32)
    data["item_type"] = fashion_mnist_data["target"].astype(int)
    return data


def load_electricity() -> pd.DataFrame:
    """
    Load electricity demand data set [1].

    [1] M. Harries. Splice-2 comparative evaluation: Electricity pricing. Technical report,
        The University of South Wales, 1999. 

    :return: electricity demand data set
    """
    electricity_data = datasets.fetch_openml(name="electricity", as_frame=True)
    data = electricity_data["data"].astype(np.float32)
    data["day"] = data["day"].astype(int)
    data["demand"] = electricity_data["target"]
    return data


def load_data_set(data_path: str) -> pd.DataFrame:
    """
    Load specified data set (<*.csv> format).

    :param data_path: data set in <*.csv> format to load
    :return: <*.csv> data set
    """
    assert (data_path is not None) and (type(data_path) == str) and (Path(data_path).is_file())
    return pd.read_csv(data_path)


def load_covid_19(country_or_region: Optional[str] = r'Austria') -> pd.DataFrame:
    """
    Load COVID-19 data set [1].

    [1] Dong E, Du H, Gardner L. An interactive web-based dashboard to track COVID-19 in real time.
        Lancet Inf Dis. 20(5):533-534. doi: 10.1016/S1473-3099(20)30120-1

    :return: COVID-19 data set
    """

    # Load data sets.
    confirmed, deceased, recovered = (
        load_data_set(f'resources/time_series_covid19_{_}_global.csv') for _ in (r'confirmed', r'deaths', r'recovered')
    )

    # Check if selected country/region is in the data.
    assert all((country_or_region in data[r'Country/Region'].values for data in (confirmed, deceased, recovered)))

    # Filter data sets and remove unused columns.
    data = pd.concat(
        data.loc[data[r'Country/Region'] == country_or_region].drop(
            [r'Province/State', r'Country/Region', r'Lat', r'Long'], axis=1
        ) for data in (confirmed, deceased, recovered)
    ).transpose()
    data.columns = (r'confirmed', r'deceased', r'recovered')

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
                                           init="random").fit_transform(data.drop(columns=target_column)), index=data.index)
        projected_data[target_column] = data[target_column]
    else:
        projected_data = pd.DataFrame(TSNE(n_components=n_components, perplexity=float(perplexity), learning_rate=200,
                                           init="random").fit_transform(data), index=data.index)
    return projected_data


def apply_k_means(k: int, data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply k-means clustering algorithm on the specified data.

    :param k: amount of clusters
    :param data: data used for clustering
    :return: predicted cluster per dataset entry
    """
    assert (type(k) == int) and (k >= 1)
    assert type(data) == pd.DataFrame
    return KMeans(n_clusters=k).fit_predict(data)


def apply_affinity_propagation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply affinity propagation clustering algorithm on the specified data.

    :param data: data used for clustering
    :return: predicted cluster per dataset entry
    """
    assert type(data) == pd.DataFrame
    return AffinityPropagation(affinity='euclidean', random_state=0).fit_predict(data)


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
    assert not (target_column is not None and targets is not None), "can only specify either 'target_column' or 'targets"
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


def plot_image(image_path: str, alpha: float = 1.0, grayscale: bool = False, rotation_angle: float = 0.0,
               flip: Optional[str] = None, cropping: Optional[Tuple[int, int, int, int]] = None, blur_radius: int = 0,
               segmentation_thresholds: Optional[Sequence[Optional[int]]] = None, **kwargs) -> None:
    """
    Visualize image and optionally apply one or more image augmentation methods.

    :param image_path: path of image to display
    :param alpha: strength of alpha channel
    :param grayscale: flag for converting image to grayscale color model
    :param rotation_angle: angle of image rotation
    :param flip: flip image either <vertical> or <horizontal>
    :param cropping: amount of pixels to crop (left, top, right, bottom)
    :param blur_radius: radius parameter of Gaussian blur effect
    :param segmentation_thresholds: thresholds for cropping channel values (lower, upper for RGB respectively)
    :param kwargs: optional keyword arguments passed to matplotlib
    """
    assert (image_path is not None) and (type(image_path) == str) and (Path(image_path).is_file())
    assert (type(alpha) in [float, int]) and (0 <= alpha <= 1)
    assert type(grayscale) == bool
    assert type(rotation_angle) in [float, int]
    assert (flip is None) or ((type(flip) == str) and (flip.strip().lower() in [r'vertical', r'horizontal']))
    assert (cropping is None) or (
            (type(cropping) == tuple) and (0 <= len(cropping) <= 4) and all(type(_) == int for _ in cropping))
    assert type(blur_radius) in [int, float]
    assert (segmentation_thresholds is None) or (all((
        type(segmentation_thresholds) in [tuple, list], 0 <= len(segmentation_thresholds) <= 6,
        all(type(_) in [type(None), int] for _ in segmentation_thresholds))))
    with Image.open(image_path) as image_raw:
        image = image_raw.convert(r'RGBA')

    # Perform image segmentation.
    image_array = np.array(image)
    if segmentation_thresholds is not None:
        for threshold_index, (channel_index, comparison) in enumerate(product(range(3), [gt, lt])):
            current_threshold = segmentation_thresholds[threshold_index]
            if current_threshold is not None:
                image_array[:, :, channel_index] *= comparison(image_array[:, :, channel_index], current_threshold)
    image = Image.fromarray(image_array)

    # Perform image blur.
    image = image.filter(ImageFilter.GaussianBlur(int(blur_radius)))

    # Perform RGBA conversion and optionally image to grayscale conversion.
    image.putalpha(int(alpha * 256))
    if grayscale:
        image = image.convert(r'LA')

    # Perform image cropping.
    if cropping is not None:
        cropping = list(cropping) + [0] * (4 - len(cropping))
        cropping[2:] = [image.size[_] - cropping[2 + _] for _ in range(2)]
        image = image.crop(cropping)

    # Perform image rotation.
    image = image.rotate(rotation_angle, expand=True)

    # Perform image flipping.
    if flip is not None:
        image = image.transpose(Image.FLIP_LEFT_RIGHT if flip.strip().lower() == r'vertical' else Image.FLIP_TOP_BOTTOM)

    # Show image.
    fig, ax = plt.subplots(**kwargs)
    ax.imshow(image)
    plt.gca().axis(r'off')
    plt.show()


def plot_image_channels_rgb(image_path: str, **kwargs) -> None:
    """
    Visualize channels of the specified image (separately).

    :param image_path: path of image to analyze
    :param kwargs: optional keyword arguments passed to matplotlib
    """
    assert (image_path is not None) and (type(image_path) == str) and (Path(image_path).is_file())
    with Image.open(image_path) as image_raw:
        image = np.array(image_raw)

    assert (len(image.shape) == 3) and (image.shape[2] == 3)
    fig, ax = plt.subplots(1, 3, **kwargs)
    for channel_index, channel_type in enumerate([r'red channel', r'green channel', r'blue channel']):
        channel_placeholder = np.zeros(image.shape, dtype=r'uint8')
        channel_placeholder[:, :, channel_index] = image[:, :, channel_index]
        ax[channel_index].imshow(channel_placeholder)
        ax[channel_index].set_title(channel_type)
        ax[channel_index].axis(r'off')
    plt.show()


def plot_image_histogram(image_path: str, **kwargs) -> None:
    """
    Visualize histograms of color channels.

    :param image_path: path of image to analyze
    :param kwargs: optional keyword arguments passed to matplotlib
    """
    assert (image_path is not None) and (type(image_path) == str) and (Path(image_path).is_file())

    with Image.open(image_path) as image_raw:
        image = np.asarray(image_raw)
    fig, (axr, axg, axb) = plt.subplots(1, 3, **kwargs)
    axr.hist(image[:, :, 0].ravel(), bins=255, histtype=r'stepfilled', density=True, color=r'red')
    axg.hist(image[:, :, 1].ravel(), bins=255, histtype=r'stepfilled', density=True, color=r'green')
    axb.hist(image[:, :, 2].ravel(), bins=255, histtype=r'stepfilled', density=True, color=r'blue')
    plt.show()


def plot_decision_boundaries(data: pd.DataFrame, classifier: ClassifierMixin, target_column: Optional[str] = None,
                             granularity: float = 10.0, legend: bool = True, **kwargs) -> None:
    """
    Visualize decision boundaries of specified classifier in a two-dimensional plot.

    :param data: data set for which to visualize decision boundaries
    :param classifier: classifier used to compute decision boundaries
    :param target_column: optional target column to be used for color-coding (defaults to last column)
    :param granularity: granularity of visualized color mesh
    :param legend: flag for displaying a legend
    :param kwargs: optional keyword arguments passed to matplotlib
    """
    assert (type(data) == pd.DataFrame) and (data.shape[1] == 3)
    assert (target_column is None) or ((type(target_column) == str) and (target_column in data))
    assert type(legend) == bool

    # Prepare data and mesh grid for plotting.
    if target_column is None:
        data_stripped = data
        hue = data[data.columns[2]]
        cmap = ListedColormap(sns.color_palette().as_hex()[:len(set(data[data.columns[2]]))])
    else:
        data_stripped = data.drop(columns=target_column)
        hue = data[target_column]
        cmap = ListedColormap(sns.color_palette().as_hex()[len(set(target_column))])
    xx, yy = np.meshgrid(np.arange(data_stripped[0].min() - 1, data_stripped[0].max() + 1, granularity),
                         np.arange(data_stripped[1].min() - 1, data_stripped[1].max() + 1, granularity))
    target = classifier.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])).astype(dtype=np.float32).reshape(xx.shape)

    # Plot color mesh of decision boundaries.
    fig, ax = plt.subplots(**kwargs)
    ax.pcolormesh(xx, yy, target, cmap=cmap, shading=r'auto')
    
    # Plot invisible auxiliary scatter plot in order to display a legend.
    if legend:
        if hasattr(cmap, r'colors'):
            cmap = cmap.colors
        sns.scatterplot(x=data_stripped.iloc[0,0], y=data_stripped.iloc[0,1], hue=hue, palette=cmap, alpha=0.0, legend=r'full')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
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
    for data, target in data_loader:
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
