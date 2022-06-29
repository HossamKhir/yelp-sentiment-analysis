from setuptools import setup

setup(
    name="sentanal_yelp",
    version="0.1.0",
    description="package of utilities used on sentiment analysis of yelp review dataset",  # noqa: E501
    author="Hossam Khair",
    author_email="h.khair95@outlook.com",
    packages=["spacy", "sentence-transformers", "numpy", "pandas", "nltk"],
)
