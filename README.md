# yelp-sentiment-analysis

Sentiment analysis on dataset from Yelp reviews

- [yelp-sentiment-analysis](#yelp-sentiment-analysis)
  - [Setup](#setup)
  - [About the dataset](#about-the-dataset)
  - [Working on the dataset](#working-on-the-dataset)
  - [Modelling](#modelling)
    - [Benchmark model](#benchmark-model)
    - [Classic models](#classic-models)
    - [Deep Learning](#deep-learning)
  - [Summarised Results](#summarised-results)
  - [Next steps](#next-steps)
  - [Licence](#licence)

## Setup

It is recommended to have a separate python environment for the repo.

> NOTE: all notebooks were run on [Kaggle](https://www.kaggle.com)

```sh

python3 -m venv .yelp_sent --system-site-packages # use py if on windows, python if on Mac
source ./.yelp_sent/bin/activate # look for equivalent for windows or Mac
python3 -m pip install -r requirements.txt

```

## About the dataset

The [Yelp Review Sentiment Dataset](https://www.kaggle.com/datasets/ilhamfp31/yelp-review-dataset) is extracted from Yelp Dataset Challange $2015$.

This is a relatively big dataset with $560$K training reviews, equally split to positive and negative by $280$K review each and $38$K test reviews

More details about the exploration of the dataset is in the [EDA](./notebooks/01.eda.ipynb) notebook.

## Working on the dataset

The dataset was preprocessed in the [Preprocess](./notebooks/02.preprocessing.ipynb) notebook. The preprocessing was simple: dropping URLs, removing stopwords, tokenising and lowercasing each review.

> TODO: using autocorrect might have some effect on the quality of the dataset

## Modelling

### Benchmark model

To start handling the task of classification, a simple [benchmark model](./notebooks/03.benchmark.ipynb) was created, a Multinomial Naive Bayes classifier, working on TF-IDF vectors from the unigram tokens from the dataset.

### Classic models

For classic models, a [Bigram Naive Bayes](./notebooks/04.bigram-naive-bayes.ipynb) variant of the benchmark model, [Logistic Regression and Support Vector Machines](./notebooks/06.classic-ml.ipynb) were employed, working on a [transformed vector representation](./notebooks/05.sentence-transform.ipynb) of the reviews of the dataset.

### Deep Learning

> TODO: consider building some FC, CNN, RNNs models.

[Pre-trained transformers](./notebooks/07.pre-trained.ipynb) were used to predict the entire training dataset.

> TODO: fine tune said transformers for better results.

## Summarised Results

The score used is the `f1-score` accuracy.

| Model                                           |  Train  | Validation | Test  |
| :---------------------------------------------- | :-----: | :--------: | :---: |
| Unigram NB                                      | $0.875$ |  $0.879$   | `TBD` |
| Bigram NB                                       | $0.903$ |  $0.906$   | `TBD` |
| Logistic Regression                             | $0.894$ |  $0.893$   | `TBD` |
| Support Vector Machine                          | $0.893$ |  $0.891$   | `TBD` |
| SpacyTextBlob<sup>\*</sup>                      |  `N/A`  |  $0.682$   | `TBD` |
| distilbert-base-uncased-finetuned-sst-2-english |  `N/A`  |  $0.907$   | `TBD` |

> \*: SpacyTextBlob is not a pre-trained classifier, but it had an attribute `polarity` that could be used to inform a classifier.

## Next steps

- [ ] Hyper-Parameter tuning for classic ML models
- [ ] Fully Connected, CNN, RNN models
- [ ] Fine-tune pre-trained transformer

## Licence

Check [LICENSE](./LICENSE) for more information
