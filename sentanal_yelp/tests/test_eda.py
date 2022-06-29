#! /usr/bin/env python3
import numpy as np
import pandas as pd
from sentanal_yelp.eda import drop_non_alnum, build_polygram, get_most_common


def test_drop_non_alnum():
    sample = "full stop. exclamation mark! question mark?"
    target = "full stop exclamation mark question mark"
    result = drop_non_alnum(sample)
    assert result == target


def test_build_polygram():
    tokens = "Once upon a time, in a far off kingdom".split()
    target_bigram = [
        "Once upon",
        "upon a",
        "a time,",
        "time, in",
        "in a",
        "a far",
        "far off",
        "off kingdom",
    ]
    target_trigram = [
        "Once upon a",
        "upon a time,",
        "a time, in",
        "time, in a",
        "in a far",
        "a far off",
        "far off kingdom",
    ]
    result = build_polygram(tokens, n=2)
    assert result == target_bigram
    result = build_polygram(tokens, n=3)
    assert result == target_trigram


def test_get_most_common():
    tokens = [
        ["!", "!", "."],
        ["@", ".", "!"],
        ["?", "!", "."],
    ]
    target = pd.DataFrame({"token": ["!", "."], "frequency": [4, 3]}).set_index("token")
    result = get_most_common(tokens, 2)
    assert np.all(result.index == target.index) and np.all(
        result.values == target.values
    )
