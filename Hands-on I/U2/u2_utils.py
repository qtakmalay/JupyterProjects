# -*- coding: utf-8 -*-
"""
Authors: Brandstetter, Schäfl, Schörgenhumer
Date: 24-10-2022

This file is part of the "Hands on AI I" lecture material. The following copyright statement applies
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
import sys
import scipy
import sklearn
import spacy

from PIL import Image
from distutils.version import LooseVersion
from IPython.core.display import HTML
from matplotlib.image import imread
from scipy import signal, ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.io.wavfile import read
from pathlib import Path
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, Tuple, Union


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
    seaborn_check = '(\u2713)' if LooseVersion(sns.__version__) >= LooseVersion('0.10.0') else '(\u2717)'
    scipy_check = '(\u2713)' if LooseVersion(scipy.__version__) >= LooseVersion(r'1.5.0') else '(\u2717)'
    spacy_check = '(\u2713)' if LooseVersion(spacy.__version__) >= LooseVersion(r'2.3.0') else '(\u2717)'
    print(f'Installed Python version: {sys.version_info.major}.{sys.version_info.minor} {python_check}')
    print(f'Installed numpy version: {np.__version__} {numpy_check}')
    print(f'Installed pandas version: {pd.__version__} {pandas_check}')
    print(f'Installed scikit-learn version: {sklearn.__version__} {sklearn_check}')
    print(f'Installed matplotlib version: {matplotlib.__version__} {matplotlib_check}')
    print(f'Installed seaborn version: {sns.__version__} {seaborn_check}')
    print(f'Installed scipy version: {scipy.__version__} {scipy_check}')
    print(f'Installed spacy version: {spacy.__version__} {spacy_check}')


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
    return AffinityPropagation(affinity='euclidean', random_state=None).fit_predict(data)


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


def plot_image(image_path: str) -> None:
    """
    Plot image for given image path.

    :param image_path: path to image
    :return: None
    """
    assert (image_path is not None) and (type(image_path) == str) and (Path(image_path).is_file())
    img = imread(image_path)
    plt.imshow(img)
    plt.gca().axis(r'off')
    plt.show()


def plot_image_channels_rgb(image_path: str) -> None:
    """
    Plot rgb image channels.

    :param image_path: path to image
    :return: None
    """
    assert (image_path is not None) and (type(image_path) == str) and (Path(image_path).is_file())
    img = imread(image_path)
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    for i, channel in enumerate([r'red channel', r'green channel', r'blue channel']):
        tmp = np.zeros(img.shape, dtype='uint8')
        tmp[:, :, i] = img[:, :, i]
        ax[i].imshow(tmp)
        ax[i].set_title(channel)
        ax[i].axis(r'off')


def plot_image_rgba(image_path: str, alpha: float = 1.0) -> None:
    """
    Plot rgba image.

    :param image_path: path to image
    :param alpha: transparency parameter
    :return: None
    """
    assert (image_path is not None) and (type(image_path) == str) and (Path(image_path).is_file())
    assert (type(alpha) == float) or (type(alpha) == int)
    with Image.open(image_path) as img:
        img = img.convert(r'RGBA')
        img.putalpha(int(alpha * 256))
        plt.imshow(img)
        plt.gca().axis(r'off')
        plt.show()


def plot_image_grayscale(image_path: str) -> None:
    """
    Plot grayscale image.

    :param image_path: path to image
    :return: None
    """
    assert (image_path is not None) and (type(image_path) == str) and (Path(image_path).is_file())
    with Image.open(image_path) as img:
        img = img.convert(r'I')
        plt.imshow(img)
        plt.gca().axis(r'off')
        plt.show()


def plot_rotated_image(image_path: str, angle: int = 45) -> None:
    """
    Plot rotated image.

    :param image_path: path to image
    :param angle: rotation angle
    :return: None
    """
    assert (image_path is not None) and (type(image_path) == str) and (Path(image_path).is_file())
    assert (type(angle) == float) or (type(angle) == int)
    img = imread(image_path)
    img_rotated = ndimage.rotate(img, angle=angle)
    plt.imshow(img_rotated)
    plt.gca().axis(r'off')
    plt.show()


def plot_flipped_image(image_path: str, flipping: str) -> None:
    """
    Plot flipped image.

    :param image_path: path to image
    :param flipping: vertical or horizontal flip
    :return: None
    """
    assert (image_path is not None) and (type(image_path) == str) and (Path(image_path).is_file())
    assert (flipping == r'vertical') or (flipping == r'horizontal')
    img = imread(image_path)
    if flipping == r'horizontal':
        plt.imshow(img[:, ::-1, :])
    elif flipping == r'vertical':
        plt.imshow(img[::-1, :, :])
    plt.gca().axis(r'off')
    plt.show()


def plot_cropped_image(image_path: str, left: int = 0, top: int = 0, width: int = 10, height: int = 10) -> None:
    """
    Plot cropped image.

    :param image_path: path to image
    :param left: pixels cropped on the left
    :param top: pixels cropped on the top
    :param width: width of the extracted image
    :param height: height of the extracted image
    :return: None
    """
    assert (image_path is not None) and (type(image_path) == str) and (Path(image_path).is_file())
    assert (type(left) == int) and (type(top) == int) and (type(width) == int) and (type(height) == int)
    with Image.open(image_path) as img:
        img_cropped = img.crop((left, top, left + width, top + height))
        plt.imshow(img_cropped)
        plt.gca().axis(r'off')
        plt.show()


def plot_blurred_image(image_path: str, sigma: float) -> None:
    """
    Plot blurred image.

    :param image_path: path to image
    :param sigma: parameter to control the standard deviation of the Gaussian filter (the higher the blurrier)
    :return: None
    """
    assert (image_path is not None) and (type(image_path) == str) and (Path(image_path).is_file())
    assert (type(sigma) == float) or (type(sigma) == int)
    img = imread(image_path)
    blurred = gaussian_filter(img, sigma=(sigma, sigma, 1))
    plt.imshow(blurred)
    plt.gca().axis('off')
    plt.show()


def plot_color_histograms(image_path: str) -> None:
    """
    Plot histograms for each color channel.

    :param image_path: path to image
    :return: None
    """
    assert (image_path is not None) and (type(image_path) == str) and (Path(image_path).is_file())
    img = imread(image_path)
    fig, (axr, axg, axb) = plt.subplots(1, 3, figsize=(15, 3))
    axr.hist(img[:, :, 0].ravel(), bins=255, histtype=r'stepfilled', density=True, color=r'red')
    axg.hist(img[:, :, 1].ravel(), bins=255, histtype=r'stepfilled', density=True, color=r'green')
    axb.hist(img[:, :, 2].ravel(), bins=255, histtype=r'stepfilled', density=True, color=r'blue')


def plot_word_embeddings_2d(data: pd.DataFrame, words: list = None, figsize: tuple = (10, 10)) -> None:
    """
    Plot word embeddings in 2 dimensions.

    :param data: dataframe of embeddings
    :param words: list of words to use instead of the dataframe index
    :param figsize: size of figure
    :return: None
    """
    assert (type(data) == pd.DataFrame) and (data.shape[1] == 2)
    assert (words is None) or ((type(words) == list) and all(isinstance(s, str) for s in words))
    assert (type(figsize) == tuple) and all((type(_) == int for _ in figsize))
    plt.figure(figsize=figsize)
    plt.margins(0.1)
    plt.scatter(data[:][0], data[:][1])
    for word, (x, y) in zip(words if words is not None else data.index, data.values):
        plt.text(x + 0.07, y, word, size=13)
    plt.show()


def plot_word_embeddings_3d(data: pd.DataFrame, words: list = None, figsize: tuple = (10, 10)) -> None:
    """
    Plot word embeddings in 3 dimensions.

    :param data: dataframe of embeddings
    :param words: list of words to use instead of the dataframe index
    :param figsize: size of figure
    :return: None
    """
    assert (type(data) == pd.DataFrame) and (data.shape[1] == 3)
    assert (words is None) or ((type(words) == list) and all(isinstance(s, str) for s in words))
    assert (type(figsize) == tuple) and all((type(_) == int for _ in figsize))
    fig = plt.figure(figsize=figsize)
    subfig = fig.add_subplot(111, projection='3d')
    plt.margins(0.1)
    subfig.scatter(data[:][0], data[:][1], data[:][2])
    for word, (x, y, z) in zip(words if words is not None else data.index, data.values):
        subfig.text(x + 0.07, y, z, word, size=13)
    plt.show()


def segment_image(image_path: str, figsize: tuple = (15, 10),
                  upper_threshold_r: int = 255, upper_threshold_g: int = 255, upper_threshold_b: int = 255,
                  lower_threshold_r: int = 0, lower_threshold_g: int = 0, lower_threshold_b: int = 0) -> None:
    """
    Segment image with upper and lower thresholds for red channel, green channel and blue channel.

    :param image_path: path to image
    :param figsize: size of figure (the height might be altered based on the image aspect ratio)
    :param upper_threshold_r: keep values <= threshold for red channel
    :param upper_threshold_g: keep values <= threshold for green channel
    :param upper_threshold_b: keep values <= threshold for blue channel
    :param lower_threshold_r: keep values >= threshold for red channel
    :param lower_threshold_g: keep values >= threshold for green channel
    :param lower_threshold_b: keep values >= threshold for blue channel
    :return: None
    """
    from PIL import Image
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        np_img = np.asarray(img)
    r_mask = (np_img[:, :, 0] <= upper_threshold_r) & (np_img[:, :, 0] >= lower_threshold_r)
    g_mask = (np_img[:, :, 1] <= upper_threshold_g) & (np_img[:, :, 1] >= lower_threshold_g)
    b_mask = (np_img[:, :, 2] <= upper_threshold_b) & (np_img[:, :, 2] >= lower_threshold_b)
    mask = ~(r_mask & g_mask & b_mask)
    masked_img = np_img.copy()
    masked_img[mask] = 255
    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    ax1.imshow(mask, cmap='Greys')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.imshow(img)
    ax2.imshow(mask, cmap='Greys', alpha=0.4)
    ax2.axis('off')
    ax3.imshow(masked_img)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)


def generate_wave(freq: float, time: float, sampling_rate: int) -> np.ndarray:
    """
    Generating sine wave.

    :param freq: wave frequency
    :param time: length of signal
    :param sampling_rate: sampling rate
    :return: array comprising the generated wave
    """
    assert (type(freq) == float) or (type(freq) == int)
    assert (type(time) == float) or (type(time) == int)
    assert (type(sampling_rate) == float) or (type(sampling_rate) == int)
    t = np.linspace(0, time, int(time * sampling_rate), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def plot_wave(points: np.ndarray, time: float, sampling_rate: int) -> None:
    """
    General function for plotting waves.

    :param points: points of wave
    :param time: length of signal
    :param sampling_rate: sampling rate
    :return: None
    """
    assert (type(points) == np.ndarray)
    assert (type(time) == float) or (type(time) == int)
    assert (type(sampling_rate) == float) or (type(sampling_rate) == int)
    t = np.linspace(0, time, int(sampling_rate * time), endpoint=False)
    plt.plot(t, points)
    plt.show()


def plot_wave_with_sampling_points(points: np.ndarray, time: float, sampling_rate: int) -> None:
    """
    Plot wave from sampling points.

    :param points: points of wave
    :param time: length of signal
    :param sampling_rate: sampling rate
    :return: None
    """
    assert (type(points) == np.ndarray)
    assert (type(time) == float) or (type(time) == int)
    assert (type(sampling_rate) == float) or (type(sampling_rate) == int)
    n_samples = int(sampling_rate * time)
    t = np.linspace(0, time, n_samples, endpoint=False)
    plt.plot(t, points)
    plt.plot(t, points, r'o')
    plt.show()


def plot_wave_with_sampling_rate(points: np.ndarray, freq: float, time: float, sampling_rate: int) -> None:
    """
    Plot wave and sampling points.

    :param points: points of the wave
    :param freq: wave frequency
    :param time: length of signal
    :param sampling_rate: sampling rate
    :return: None
    """
    assert (type(points) == np.ndarray)
    assert (type(freq) == float) or (type(freq) == int)
    assert (type(time) == float) or (type(time) == int)
    assert (type(sampling_rate) == float) or (type(sampling_rate) == int)
    t1 = np.linspace(0, time, len(points))
    plt.plot(t1, points)
    # create equidistant sampling times
    n_samples = np.ceil(sampling_rate * time).astype(int)
    t2 = [i / sampling_rate for i in range(n_samples)]
    if n_samples > len(points):
        # if there are more samples than there are points, simply create a new wave instead of actual sampling
        sampled_points = generate_wave(freq, time, sampling_rate)
    else:
        # otherwise, create the index values to retrieve the actual data points
        # we only want samples from "points" up until the last sampling time from "t2", so we
        # skip the remainder of the input points by specifying "end" of "linspace" accordingly
        idx = np.round(np.linspace(0, (len(points) / time) * t2[-1], n_samples, endpoint=True)).astype(int)
        sampled_points = points[idx]
    plt.plot(t2, sampled_points, 'o')
    plt.show()


def apply_fourier_transform(points: np.ndarray) -> np.ndarray:
    """
    Apply Fourier Transform to points of wave.

    :param points: points of the wave
    :return: result of Fourier transformation as pd.Series (the index contains the
             Discrete Fourier Transform sample frequencies)
    """
    assert (type(points) == np.ndarray)
    points_fourier_transform = scipy.fft.rfft(points)
    winsize = len(points_fourier_transform) * 2 - 2
    freqs = scipy.fft.fftfreq(winsize)[:len(points_fourier_transform)]
    data = 2 * np.abs(points_fourier_transform) / len(points)
    return pd.Series(data, index=freqs)


def plot_spectrum(ft_points: Union[np.ndarray, pd.Series], sampling_rate: int, max_freq: float = None) -> None:
    """
    Plot magnitude spectrum.

    :param ft_points: magnitude spectrum bins
    :param sampling_rate: sampling rate
    :param max_freq: optionally only plot spectrum up to this frequency
    """
    assert isinstance(ft_points, (np.ndarray, pd.Series))
    assert isinstance(sampling_rate, (int, float))
    assert (max_freq is None or isinstance(max_freq, (int, float)))
    assert (max_freq is None or max_freq < sampling_rate / 2), "max_freq can be at most half the sampling_rate"
    if isinstance(ft_points, pd.Series):
        ft_points = ft_points.values
    winsize = len(ft_points) * 2 - 2
    freqs = scipy.fft.fftfreq(winsize)[:len(ft_points)] * sampling_rate
    if max_freq:
        max_bin = np.searchsorted(freqs, max_freq)
        freqs = freqs[:max_bin]
        ft_points = ft_points[:max_bin]
    plt.plot(freqs, ft_points)
    plt.show()


def read_wav_file(wav_file: str, time: int = None) -> Tuple[np.ndarray, int]:
    """
    Read in sound file.

    :param wav_file: path to sound file
    :param time: length of sound signal
    :return: data signal and corresponding signal rate
    """
    assert all((wav_file is not None, type(wav_file) == str, Path(wav_file).is_file(), wav_file.endswith('.wav')))
    assert (time is None) or (type(time) == float) or (type(time) == int)
    file = read(wav_file)
    signal_rate = file[0]
    points = file[1]
    # check for multiple channels (should only be stereo, i.e., 2 at max)
    if len(points.shape) > 1:
        points = points[:, 0]  # just take the first channel
    if time:
        length_of_signal = int(time * signal_rate)
        if length_of_signal < len(points):
            points = points[0:length_of_signal]
    return points, signal_rate


def compute_spectrogram(points: np.ndarray, winsize: int = 1024, hopsize: int = 512) -> np.ndarray:
    """
    Compute magnitude spectrogram.

    :param points: audio samples
    :param winsize: FFT window size
    :param hopsize: distance between starting points of FFT windows
    """
    assert isinstance(points, np.ndarray)
    assert isinstance(winsize, int)
    assert isinstance(hopsize, int)
    num_frames = (len(points) - winsize) // hopsize + 1
    frames = np.lib.stride_tricks.as_strided(
            points, shape=(num_frames, winsize),
            strides=(points.strides[0] * hopsize, points.strides[0]))
    window = np.hanning(winsize).astype(points.dtype)
    spect = scipy.fft.rfft(frames * window)
    magspect = np.abs(spect)
    return magspect


def plot_spectrogram(spectrogram: np.ndarray, sampling_rate: int, max_freq: float = None, show_values: bool = True) -> None:
    """
    Plot magnitude spectrogram.

    :param spectrogram: magnitude spectrogram
    :param sampling_rate: sampling rate
    :param max_freq: optionally only plot spectrogram up to this frequency
    :param show_values: whether to display a color bar showing the values/amplitudes of the frequencies
    """
    assert isinstance(spectrogram, np.ndarray)
    assert isinstance(sampling_rate, (int, float))
    assert (max_freq is None or isinstance(max_freq, (int, float)))
    assert (max_freq is None or max_freq < sampling_rate / 2), "max_freq can be at most half the sampling_rate"
    num_bins = spectrogram.shape[-1]
    winsize = num_bins * 2 - 2
    freqs = scipy.fft.fftfreq(winsize)[:num_bins] * sampling_rate
    if max_freq:
        max_bin = np.searchsorted(freqs, max_freq)
        spectrogram = spectrogram[..., :max_bin]
    else:
        max_freq = freqs[-1]
    plt.imshow(np.maximum(1e-3, spectrogram.T), origin='lower', cmap='gray', interpolation='none',
               extent=(-0.5, len(spectrogram) - 0.5, 0, max_freq),
               norm=matplotlib.colors.LogNorm(), aspect='auto')
    plt.xlabel('time frame')
    plt.ylabel('freq (Hz)')
    if show_values:
        plt.colorbar()
    plt.show()


def convert_to_onehot(vocab: list, sentence: str) -> pd.DataFrame:
    """
    Convert input to one-hot encoding.

    :param vocab: vocabulary
    :param sentence: input sentence
    :return: data frame comprising one-hot encoded input
    """
    assert (type(vocab) == list)
    assert (type(sentence) == str)
    words_sentence = sentence.split()
    words_vocab = np.unique(vocab)
    one_hot = np.zeros(shape=(len(words_sentence), len(words_vocab)), dtype=np.uint8)
    for i, word in enumerate(words_sentence):
        try:
            one_hot[i, vocab.index(word)] = 1
        except Exception as e:
            print(e)
    return pd.DataFrame(data=one_hot, index=words_sentence)


def get_word_vectors(word_embedding_english: spacy.lang, words: list) -> pd.DataFrame:
    """
    Get embedding vectors for list of words.

    :param word_embedding_english: dataset of embeddings
    :param words: list of words to embed
    :return: data frame of embedded words
    """
    assert (type(word_embedding_english) == spacy.lang.en.English)
    assert (type(words) == list) and all(isinstance(s, str) for s in words)
    return pd.DataFrame([word_embedding_english(word).vector for word in words], index=words)


def find_similar_words(query: np.ndarray, data: pd.DataFrame):
    """
    Find similar and dissimilar words for query in word embeddings

    :param query: query embedding vector
    :param data: dataframe of embeddings
    :return: list of similar words, list of dissimilar words
    """
    assert (type(query) == np.ndarray)
    assert (type(data) == pd.DataFrame)
    norms = data.apply(np.square).sum(axis=1).apply(np.sqrt)
    data = data.apply(lambda x: x / norms).dot(query).sort_values()
    similar = data.index[:-4:-1].to_list()
    distant = data.index[:3].to_list()
    return similar, distant
