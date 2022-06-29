#! /usr/bin/env python3
from sentanal_yelp.preprocess import preprocess


def test_preprocess():
    doc = "My name is Barry Allen, & I'm the fastest man alive!\nhttps://alwayslate.co"
    target = "my barry allen fast man alive"
    result = preprocess(doc)
    assert result == target
