"""
"""
import re
from itertools import chain
from typing import Union

import nltk
import numpy as np
import pandas as pd
from nltk.probability import FreqDist
from nltk.util import ngrams

nltk.download("punkt")


def drop_non_alnum(string: str) -> str:
    """replaces any non alphanumeric character with a space,
    reduces multiple consecutive spaces into 1 space,
    converts a string of only whitespace into empty string

    Parameters
    ----------
    string: str :
        the input text to be processed


    Returns
    -------
    out: str
        the processed text, it might be an empty string


    """
    # replace each non-alphanumeric into a space
    string = re.sub(r"\W", " ", string)
    # replace multiple whitespace characters into a single one
    string = re.sub(r"\s+", " ", string).strip()
    # replace whitespace only strings w/ empty strings
    return re.sub(r"^\s+$", "", string)


def get_most_common(
    tokens: Union[list, np.ndarray, pd.Series], n: int = 20
) -> pd.DataFrame:
    """Returns the most common `n` tokens within the given `tokens`

    Parameters
    ----------
    tokens: Union[list :
    np.ndarray :
    pd.Series] :
        a collection of tokens, or a collection of collection of tokens
    n: int :
         (Default value = 20)
         the number of the top most common to look for

    Returns
    -------
    out: pandas.DataFrame
        a pandas data frame, of shape (n, 1), the index is the token, and the
        column is the frequency of the token


    """
    # get a single list of tokens,
    all_tokens = chain.from_iterable(tokens)
    # get a counter dict from tokens
    token_count = FreqDist(all_tokens)
    # return top n most common
    most_common = pd.DataFrame(token_count.most_common(n))
    most_common.columns = ["token", "frequency"]
    return most_common.set_index("token")


def build_polygram(tokens: Union[list, np.ndarray], n: int = 2) -> list:
    """from a list of tokens, build a list of n-polygrams

    Parameters
    ----------
    tokens: Union[list :
    np.ndarray] :
        a collection of tokens
    n: int :
         (Default value = 2)
        the number of grams in ngrams, 2 for bigrams, 3 for trigrams, and so on

    Returns
    -------
    out: list
        a list of the ngrams needed


    """
    return [" ".join(gram) for gram in ngrams(tokens, n)]
