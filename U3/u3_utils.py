# -*- coding: utf-8 -*-
"""
Authors: Brandstetter, Rumetshofer, Parada-Cabaleiro, Schörgenhumer
Date: 07-11-2022

This file is part of the "Hands on AI I" lecture material. The following copyright statement applies
to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for personal and non-commercial
educational use only. Any reproduction of this manuscript, no matter whether as a whole or in parts, no matter whether
in printed or in electronic form, requires explicit prior acceptance of the authors.
"""
import math
import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from distutils.version import LooseVersion
from IPython.core.display import HTML
from sklearn import datasets, clone
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import Optional, Union, Sequence


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
    print(f'Installed Python version: {sys.version_info.major}.{sys.version_info.minor} {python_check}')
    print(f'Installed numpy version: {np.__version__} {numpy_check}')
    print(f'Installed pandas version: {pd.__version__} {pandas_check}')
    print(f'Installed scikit-learn version: {sklearn.__version__} {sklearn_check}')
    print(f'Installed matplotlib version: {matplotlib.__version__} {matplotlib_check}')
    print(f'Installed seaborn version: {sns.__version__} {seaborn_check}')

    
def plot_function(x, y, function):
    """
    Plot data points sampled from a function and the underlying function

    :param x: data points 
    :param y: function value for the data points x
    :param function: method that defines the function taking x as parameter
    """
    assert x is not None and y is not None and function is not None
    assert len(x) > 0
    assert len(x) == len(y)
    
    plt.figure(figsize=(6.5, 5))    
    # plot function
    x_test = np.linspace(0, 1, 1000)
    plt.plot(x_test, function(x_test), label="True function")
    # plot points defined by x and y
    plt.scatter(x, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    # define plot range
    plt.xlim((0, 1))
    plt.ylim((-2.2, 2.2))
    plt.legend(loc="best")
    plt.show()    
    

def plot_polynomial_fit(x, y, function, degrees, ncols=2, figsize=None):
    """
    Fit polynomials to the data points x,y and plot the results 
    together with the points and the original function

    :param x: data points 
    :param y: function value for the data points x
    :param function: method that defines the function taking x as parameter
    :param degrees: degrees for the polynomial functions (list of degrees or single degree)
    :param ncols: number of subplot columns
    :param figsize: size of the entire subplot
    """
    assert x is not None and y is not None and function is not None
    assert len(x) > 0
    assert len(x) == len(y)
    assert (type(degrees) in (list, tuple) and len(degrees) >= 1) or isinstance(degrees, int)

    if isinstance(degrees, int):
        degrees = [degrees]

    # make a copy since we change the data below
    x = np.array(x)

    ncols = min(len(degrees), ncols)
    nrows = math.ceil(len(degrees) / ncols)
    # just estimate a decent figure size
    if figsize is None:
        figsize = (ncols * 7.5, nrows * 5)

    # plot function, points defined by x and y
    # and polynomials fitting to the points
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    for i, degree in enumerate(degrees):
        col = i % ncols
        row = i // ncols
        ax = axes[row, col]
        ax.set_title(f"Polynomial of degree {degrees[i]}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim((0, 1))
        ax.set_ylim((-2.2, 2.2))
        
        # fit polynomial to points
        polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                             ("linear_regression", linear_regression)])
        pipeline.fit(x[:, np.newaxis], y)
        
        # plot everything
        x_test = np.linspace(0, 1, 100)
        ax.plot(x_test, function(x_test), label="True function")
        ax.plot(x_test, pipeline.predict(x_test[:, np.newaxis]), label="Model")
        ax.scatter(x, y, edgecolor='b', s=20, label="Samples")
        
        # add legend
        ax.legend(loc="best")
    plt.show()    
    

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
    

def apply_pca(n_components: int, data: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
    """
    Apply principal component analysis (PCA) on specified dataset and down-project data accordingly.

    :param n_components: amount of (top) principal components involved in down-projection
    :param data: dataset to down-project
    :param target_column: if specified, append target column to resulting, down-projected dataset
    :return: down-projected dataset
    """
    assert (type(n_components) == int) and (n_components >= 1)
    assert type(data) == pd.DataFrame
    assert ((type(target_column) == str) and (target_column in data)) or (target_column is None)
    if target_column is not None:
        projected_data = pd.DataFrame(PCA(n_components=n_components).fit_transform(data.drop(columns=target_column)), index=data.index)
        projected_data[target_column] = data[target_column]
    else:
        projected_data = pd.DataFrame(PCA(n_components=n_components).fit_transform(data), index=data.index)
    return projected_data


def plot_features(*, data: pd.DataFrame = None, X: pd.DataFrame = None, y: Union[list, np.ndarray, pd.Series] = None,
                  features: Sequence[str] = None, target_column: Optional[str] = None, sns_kwargs: dict = None) -> None:
    """
    Visualizes the specified features of the dataset via pairwise relationship plots. Optionally,
    the displayed data points can be colored according to the specified ``target_column``.

    :param data: dataset containing the features and labels (must be specified if ``X`` and ``y`` are not used)
    :param X: features (must be specified together with ``y`` if ``data`` is not used)
    :param y: labels (must be specified together with ``X`` if ``data`` is not used)
    :param features: the list of features to visualize
    :param target_column: if specified, color the visualized data points according to this target (if ``X`` and
    ``y`` are specified, ``y`` is automatically assumed to be the target)
    :param sns_kwargs: additional keyword arguments that are passed to ``sns.pairplot`` (must not
        contain any of "data", "vars", "hue")
    """
    if data is None:
        assert X is not None and y is not None and target_column is None
        if not isinstance(y, pd.Series):
            assert "target_column" not in X.columns
            y = pd.Series(y, name="target_column")
        data = pd.concat([X, y], axis=1)
        target_column = data.columns[-1]
    else:
        assert X is None and y is None
    if features is None:
        features = data.columns
    elif isinstance(features, str):
        features = [features]
    if sns_kwargs is None:
        sns_kwargs = dict(palette="deep")
    sns.pairplot(data=data, vars=features, hue=target_column, **sns_kwargs)


def plot_points_2d(data: pd.DataFrame, target_column: Optional[str] = None, legend: bool = True,
                   sns_kwargs: dict = None, **kwargs) -> None:
    """
    Visualize data points in a two-dimensional plot, optionally colored according to ``target_column``.

    :param data: dataset to visualize
    :param target_column: optional target column to be used for color-coding
    :param legend: flag for displaying a legend
    :param sns_kwargs: additional keyword arguments that are passed to ``sns.scatterplot`` (must not
        contain any of "data", "x", "y", "hue", "legend", "ax)
    :param kwargs: keyword arguments that are passed to ``plt.subplots``
    """
    assert (type(data) == pd.DataFrame) and (data.shape[1] in [2, 3])
    assert (target_column is None) or ((data.shape[1] == 3) and (data.columns[2] == target_column))
    assert type(legend) == bool
    assert sns_kwargs is None or isinstance(sns_kwargs, dict)
    if legend:
        legend = "auto"
    if sns_kwargs is None:
        sns_kwargs = dict(palette="deep")
    _, ax = plt.subplots(**kwargs)
    sns.scatterplot(data=data, x=0, y=1, hue=target_column, legend=legend, ax=ax, **sns_kwargs)
    ax.set_xlabel(None)
    ax.set_ylabel(None)


def test_k_range(X_train: pd.DataFrame, y_train: Union[list, np.ndarray, pd.Series],
                 X_test: pd.DataFrame, y_test: Union[list, np.ndarray, pd.Series],
                 k_range, plot_train: bool = True):
    """
    Fit k-NN for different k and plot the train + test accuracies

    :param X_train: training features
    :param y_train: training labels
    :param X_test: test features
    :param y_test: test labels
    :param k_range: range of k that will be evaluated 
    :param plot_train: whether to also plot the training accuracy
    """
    plt.figure()
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        scores.append((knn.score(X_test, y_test), knn.score(X_train, y_train)))

    if not plot_train:
        scores = np.asarray(scores)[:, 0]
    plt.plot(k_range, scores, 'o')
    plt.xticks(list(k_range))
    plt.xlabel('k')
    plt.ylabel('accuracy' if plot_train else 'test accuracy')
    if plot_train:
        plt.legend(('test', 'train'))
    plt.show()


def plot_decision_boundaries(classifier, X: pd.DataFrame, y: Union[list, np.ndarray, pd.Series],
                             feature_pairs: list, ncols=3, figsize=None):
    """
    Plot the decision boundaries for the given classifier by spanning a grid on the data
    and generating a prediction for each point in the grid. Additionally, overlay the original data.

    :param classifier: a classifier from sklearn implementing the predict function
    :param X: features
    :param y: labels
    :param feature_pairs: list of string pairs (must be 2-dimensional) representing the features
    to use for fitting the classifier, or, for convenience, a pair of strings can be specified directly
    :param ncols: number of subplot columns
    :param figsize: size of the entire subplot
    """
    if isinstance(feature_pairs[0], str):
        feature_pairs = [feature_pairs]
    ncols = min(len(feature_pairs), ncols)
    nrows = math.ceil(len(feature_pairs) / ncols)
    # just estimate a decent figure size
    if figsize is None:
        figsize = (ncols * 5, nrows * 5)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    for i, feature_pair in enumerate(feature_pairs):
        col = i % ncols
        row = i // ncols
        subplot_classifier_regions(classifier, X, y, feature_names=feature_pair, axis=axes[row, col])
    plt.tight_layout()
    plt.show()


def subplot_classifier_regions(classifier, X: pd.DataFrame, y: Union[list, np.ndarray, pd.Series],
                               feature_names: list, axis) -> None:
    """
    Plot the decision boundaries for the given classifier by spanning a grid on the data
    and generating a prediction for each point in the grid. Additionally, overlay the original data.
    
    :param classifier: a classifier from sklearn implementing the predict function
    :param X: features
    :param y: labels
    :param feature_names: names of selected features (must be 2-dimensional)
    :param axis: the matplotlib axis object to use for plotting
    """
    assert (len(feature_names) == 2)

    def adjust_lightness(color, amount=0.7):
        import matplotlib.colors as mc
        import colorsys
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

    X_mat = X[feature_names].values
    y_mat = y if isinstance(y, pd.Series) else pd.Series(y, "Class")
    classes_sorted = sorted(y_mat.unique())

    # define color scheme; the number of classes determines the number of colors
    # and the order is sorted according to the class labels; this sorting is
    # necessary because the colors using "cmap=ListedColormap(...)" below will
    # use a sorting order as well and we have to match this same sorting order
    cmap = matplotlib.cm.get_cmap('Set3')
    color_list_light = [cmap(i) for i in range(len(classes_sorted))]
    color_list_dark = [adjust_lightness(c) for c in color_list_light]
    class_to_dark_color = dict(zip(classes_sorted, color_list_dark))

    # set up grid of points defined by range of x and y
    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    x2, y2 = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # get predictions of the classifier for each point in the grid
    clf = clone(classifier)
    clf.fit(X_mat, y_mat)
    pred = clf.predict(np.c_[x2.ravel(), y2.ravel()]).reshape(x2.shape)

    # plot decision boundaries
    axis.pcolormesh(x2, y2, pred, cmap=ListedColormap(color_list_light))

    # plot training points
    axis.scatter(X_mat[:, 0], X_mat[:, 1], s=50, c=y, cmap=ListedColormap(color_list_dark), edgecolor='black')
    axis.set_xlim(x2.min(), x2.max())
    axis.set_ylim(y2.min(), y2.max())

    # add legend
    legend_handles = []
    for cls, col in class_to_dark_color.items():
        patch = mpatches.Patch(color=col, label=f"{y_mat.name}: {cls}")
        legend_handles.append(patch)
    axis.legend(loc=0, handles=legend_handles)

    # set labels
    axis.set_xlabel(feature_names[0])
    axis.set_ylabel(feature_names[1])
