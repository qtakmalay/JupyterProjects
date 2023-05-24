# -*- coding: utf-8 -*-
"""
Author: N. Rekabsaz, B. Schäfl, S. Lehner, J. Brandstetter, E. Kobler, A. Schörgenhumer
Date: 16-05-2023

This file is part of the "Hands-on AI II" lecture material. The following copyright statement applies
to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for personal and non-commercial
educational use only. Any reproduction of this manuscript, no matter whether as a whole or in parts, no matter whether
in printed or in electronic form, requires explicit prior acceptance of the authors.
"""
import os
import sys
from distutils.version import LooseVersion

import numpy as np
import pandas as pd
import torch
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
    print(f'Installed Python version: {sys.version_info.major}.{sys.version_info.minor} {python_check}')
    print(f'Installed numpy version: {np.__version__} {numpy_check}')
    print(f'Installed pandas version: {pd.__version__} {pandas_check}')
    print(f'Installed PyTorch version: {torch.__version__} {pytorch_check}')


class Dictionary(object):
    """
    Dictionary class to create a dictionary based on training data
    """
    
    def __init__(self):
        self.word2idx = {'<oov>': 0}
        self.idx2word = ['<oov>']
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    """
    Corpus class to map the data of all the corpora from words to wordIDs
    """
    
    def __init__(self, path):
        assert os.path.exists(path)
        self.path = path
    
    def fill_dictionary(self, dictionary):
        # Add words to the dictionary
        with open(self.path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    dictionary.add_word(word)
    
    def words_to_ids(self, dictionary):
        # Tokenize file content
        with open(self.path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    if word in dictionary.word2idx:
                        ids.append(dictionary.word2idx[word])
                    else:
                        ids.append(dictionary.word2idx['oov'])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        
        return ids


def batchify(data, batch_size, device):
    """
    Reshape specified 1D data into a two-dimensional tensor given a batch size.
    More formally, input shape (N) is transformed into (N // B, B), where N is
    the number of elements in ``data`` and B is ``batch_size``. If N is not
    divisible without remainder, the remaining elements are dropped.
    
    :param data: The 1D data to reshape
    :param batch_size: The size of the new, second dimension
    :param device: The device where the resulting 2D tensor will be stored
    :return: The reshaped 2D data tensor
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(hidden):
    """Wraps hidden states in new Tensors, to detach them from their history.
    Starting each batch, we detach the hidden state from how it was previously produced.
    If we didn't, the model would try backpropagating all the way to start of the dataset.
    """
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(v) for v in hidden)


def get_batch(source, index: int, max_seq_len: int):
    """
    Retrieve new batch.

    :param source: Entire source data to extract the batch from
    :param index: The index position indicating the start of the batch
    :param max_seq_len: The length of the extracted batch
    :return: data, target (same as ``data`` but shifted by +1 index to the right)
    """
    seq_len = min(max_seq_len, len(source) - 1 - index)
    data = source[index:index + seq_len]
    target = source[index + 1:index + 1 + seq_len]
    return data, target


def evaluate(model: torch.nn.Module, dictionary: Dictionary,
             max_seq_len: int, eval_batch_size: int, eval_data_batches,
             criterion: torch.nn.Module = torch.nn.NLLLoss()):
    """
    Evaluate the specified model. Evaluation mode turned on to disable dropout.
    :return: Evaluation loss
    """
    model.eval()
    total_loss = 0.
    ntokens = len(dictionary)
    start_hidden = None
    
    with torch.no_grad():
        for i in range(0, eval_data_batches.size(0) - 1, max_seq_len):
            data, targets = get_batch(eval_data_batches, i, max_seq_len)
            
            if start_hidden is not None:
                start_hidden = repackage_hidden(start_hidden)
            
            output, last_hidden = model(data, start_hidden)
            total_loss += len(data) * criterion(output.view(-1, ntokens), targets.view(-1)).item()
            
            start_hidden = last_hidden
    
    return total_loss / (len(eval_data_batches) - 1)
