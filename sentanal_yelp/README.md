# sentanal_yelp

A python package with utility functions to help with exploring and preprocessing of the [Yelp Review Sentiment Dataset](https://www.kaggle.com/datasets/ilhamfp31/yelp-review-dataset)

## Installation

This can be installed using python's `pip`

```sh
python3 -m venv .my-env
source ./.my-env/bin/activate
python3 -m pip install -e sentanal_yelp

```

## Usage Example

```python

from sentanal_yelp.eda import drop_non_alum

doc = "Have u listened to @fictional's latest podcast? it is 2day fresh!"
res = drop_non_alum(doc)
print(res)  # prints: Have u listened to fictional s latest podcast it is 2day fresh

```

## Licence

This package is released under GNU General Publication License V3 or later, consult [LICENSE](./LICENSE) for more info
