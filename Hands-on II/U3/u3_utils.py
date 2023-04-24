# -*- coding: utf-8 -*-
"""
Authors: B. Schäfl, S. Lehner, J. Schimunek, J. Brandstetter, A. Schörgenhumer
Date: 18-04-2023

This file is part of the "Hands-on AI II" lecture material. The following copyright statement applies
to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for personal and non-commercial
educational use only. Any reproduction of this manuscript, no matter whether as a whole or in parts, no matter whether
in printed or in electronic form, requires explicit prior acceptance of the authors.
"""
import warnings
warnings.filterwarnings(action=r'ignore', category=UserWarning)

import multiprocessing
import sys
from distutils.version import LooseVersion
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import PIL
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit
import seaborn as sns
import sklearn
from IPython.core.display import HTML
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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
    sklearn_check = '(\u2713)' if LooseVersion(sklearn.__version__) >= LooseVersion('1.2') else '(\u2717)'
    matplotlib_check = '(\u2713)' if LooseVersion(matplotlib.__version__) >= LooseVersion('3.2.0') else '(\u2717)'
    seaborn_check = '(\u2713)' if LooseVersion(sns.__version__) >= LooseVersion('0.10.0') else '(\u2717)'
    pil_check = '(\u2713)' if LooseVersion(PIL.__version__) >= LooseVersion('6.0.0') else '(\u2717)'
    rdkit_check = '(\u2713)' if LooseVersion(rdkit.__version__) >= LooseVersion('2020.03.4') else '(\u2717)'
    print(f'Installed Python version: {sys.version_info.major}.{sys.version_info.minor} {python_check}')
    print(f'Installed numpy version: {np.__version__} {numpy_check}')
    print(f'Installed pandas version: {pd.__version__} {pandas_check}')
    print(f'Installed scikit-learn version: {sklearn.__version__} {sklearn_check}')
    print(f'Installed matplotlib version: {matplotlib.__version__} {matplotlib_check}')
    print(f'Installed seaborn version: {sns.__version__} {seaborn_check}')
    print(f'Installed PIL version: {PIL.__version__} {pil_check}')
    print(f'Installed rdkit version: {rdkit.__version__} {rdkit_check}')


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
    return KMeans(n_clusters=k, n_init="auto").fit_predict(data)


def apply_affinity_propagation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply affinity propagation clustering algorithm on the specified data.

    :param data: data used for clustering
    :return: predicted cluster per dataset entry
    """
    assert type(data) == pd.DataFrame
    return AffinityPropagation(affinity='euclidean', random_state=0).fit_predict(data)

    
def _compute_ecfps_ecfp_worker(smiles: str, radius: int) -> Dict[int, int]:
    """
    Compute ECFP of a SMILES representation.
    
    :param smiles: SMILES representation for which to compute the ECFP
    :param radius: radius to be used for each atom to compute the ECFP
    :return: ECFP of the specified SMILES representation
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f'Error parsing SMILES "{smiles}".')
    return AllChem.GetMorganFingerprint(molecule, radius).GetNonzeroElements()


def _compute_ecfps_fold_worker(ecfp: Dict[int, int], lookup: Dict[int, int], fold: int) -> np.ndarray:
    """
    Expand and fold an ECFP using a specified lookup table.
    
    :param ecfp: ECFP to expand and fold
    :param lookup: lookup table to be used to fold ECFP
    :param fold: minimum length of expanded and folded ECFP
    :return: expanded and folded ECFP
    """
    resulting_fold = max(fold, max(ecfp.values()))
    result = np.zeros(shape=(1, resulting_fold), dtype=bool)
    for key, value in ecfp.items():
        result[0, lookup[key]] = True
    return result
    

def compute_ecfps(smiles: Sequence[str], radius: int = 3, fold: int = 1024, num_jobs: int = 0) -> pd.DataFrame:
    """
    Compute ECFPs of specified SMILES representations.
    
    :param smiles: SMILES representations for which to compute the ECFPs
    :param radius: radius to be used for each atom to compute the ECFPs
    :param fold: minimum length of expanded and folded ECFPs
    :param num_jobs: amount of parallel processes to be used for computing ECFPs (<= 0: all available cores)
    :return: computed ECFPs, expanded and folded
    """
    assert (type(smiles) in (list, tuple)) and (len(smiles) > 0)
    assert all((type(_) == str) for _ in smiles)
    assert (type(radius) == int) and (radius >= 0)
    assert (type(fold) == int) and (fold > 0)
    assert type(num_jobs) == int
    
    # Compute ECFPs of specified SMILES representations in a multi-processing manner.
    with multiprocessing.Pool(multiprocessing.cpu_count() if num_jobs <= 0 else num_jobs) as worker_pool:
        ecfps = worker_pool.map(partial(_compute_ecfps_ecfp_worker, radius=radius), smiles)
    
        # Parse computed ECFPs.
        fold_mapping = {key: key % fold for key in set.union(*(set(ecfp.keys()) for ecfp in ecfps))}
        ecfps = worker_pool.map(partial(_compute_ecfps_fold_worker, lookup=fold_mapping, fold=fold), ecfps)
    
    # Combine computed ECFPs.
    ecfps = np.concatenate(ecfps)
    return pd.DataFrame(ecfps)


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
