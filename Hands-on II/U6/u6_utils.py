# -*- coding: utf-8 -*-
"""
Authors: B. Schäfl, S. Lehner, J. Schimunek, J. Brandstetter, A. Schörgenhumer
Date: 13-06-2023

This file is part of the "Hands on AI II" lecture material. The following copyright statement applies
to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for personal and non-commercial
educational use only. Any reproduction of this manuscript, no matter whether as a whole or in parts, no matter whether
in printed or in electronic form, requires explicit prior acceptance of the authors.
"""
import sys
import time
from distutils.version import LooseVersion
from typing import Any, Dict, Sequence

import gymnasium as gym
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython import display
from IPython.core.display import HTML


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
    pytorch_check = '(\u2713)' if LooseVersion(torch.__version__) >= LooseVersion('1.6') else '(\u2717)'
    matplotlib_check = '(\u2713)' if LooseVersion(matplotlib.__version__) >= LooseVersion('3.2.0') else '(\u2717)'
    seaborn_check = '(\u2713)' if LooseVersion(sns.__version__) >= LooseVersion('0.10.0') else '(\u2717)'
    gym_check = '(\u2713)' if LooseVersion(gym.__version__) >= LooseVersion('0.28.1') else '(\u2717)'
    print(f'Installed Python version: {sys.version_info.major}.{sys.version_info.minor} {python_check}')
    print(f'Installed numpy version: {np.__version__} {numpy_check}')
    print(f'Installed pandas version: {pd.__version__} {pandas_check}')
    print(f'Installed PyTorch version: {torch.__version__} {pytorch_check}')
    print(f'Installed matplotlib version: {matplotlib.__version__} {matplotlib_check}')
    print(f'Installed seaborn version: {sns.__version__} {seaborn_check}')
    print(f'Installed gym version: {gym.__version__} {gym_check}')


def set_environment_seed(environment: gym.Env, seed: int = 42) -> None:
    """
    Set seed of specified environment.

    :param environment: environment for which to set the seed
    :param seed: seed to apply
    """
    assert isinstance(environment, gym.Env)
    assert type(seed) == int and seed >= 0
    
    environment.action_space.seed(seed=seed)
    environment.observation_space.seed(seed=seed)
    environment.reset(seed=seed)
    np.random.seed(seed=seed)
