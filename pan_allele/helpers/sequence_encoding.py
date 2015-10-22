###from sequence_encoded.py

import numpy as np

def _build_index_dict(sequences):
    unique_symbols = set()
    for seq in sequences:
        for c in seq:
            unique_symbols.add(c)
    return {c: i for (i, c) in enumerate(unique_symbols)}

def sequences_to_indices(
        sequences,
        index_dict=None,
        add_start_symbol=True,
        add_end_symbol=True):
    """
    Encode sequences of symbols as sequences of integer indices starting from 1.
    Parameters
    ----------
    sequences : list of str
    index_dict : dict
        Mapping from symbols to indices (expected to start from 0)
    add_start_symbol : bool
    add_end_symbol : bool
    """
    if index_dict is None:
        index_dict = _build_index_dict(sequences)

    index_sequences = []
    for seq in sequences:
        index_sequences.append([index_dict[c] for c in seq])
    max_value = max(max(sequence) for sequence in index_sequences)
    if add_start_symbol:
        prefix = [max_value]
        max_value += 1
        index_sequences = [prefix + seq for seq in index_sequences]
    if add_end_symbol:
        suffix = [max_value]
        index_sequences = [seq + suffix for seq in index_sequences]
    return index_sequences

def padded_indices(
        sequences,
        index_dict=None,
        ndim=2,
        add_start_symbol=True,
        add_end_symbol=True):
    """
    Given a list of strings, construct a list of index sequences
    and then pad them to make an array.
    """
    index_sequences = sequences_to_indices(
        sequences=sequences,
        index_dict=index_dict,
        add_start_symbol=add_start_symbol,
        add_end_symbol=add_end_symbol)

    max_len = max(len(s) for s in index_sequences)
    n_samples = len(index_sequences)
    if ndim < 2:
        raise ValueError("Padded input must have at least 2 dims")

    shape = (n_samples, max_len) + (1,) * (ndim - 2)
    result = np.zeros(shape, dtype=int)
    for i, x in enumerate(index_sequences):
        result[i, :len(x)] = x
    return result

def onehot(sequences, index_dict=None):
    """
    Parameters
    ----------
    sequences : list of strings
    index_dict : dict
        Mapping from symbols to integer indices
    """
    n_seq = len(sequences)
    if index_dict is None:
        index_dict = _build_index_dict(sequences)
    n_symbols = len(index_dict)
    maxlen = max(len(seq) for seq in sequences)
    result = np.zeros((n_seq, maxlen, n_symbols), dtype=bool)
    for i, seq in enumerate(sequences):
        for j, sj in enumerate(seq):
            result[i, j, index_dict[sj]] = 1
    return result
