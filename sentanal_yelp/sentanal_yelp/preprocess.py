"""
"""
import re

import spacy
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess

REGEX_URL = r"(?:https?://|www\.)\S+"

nlp = spacy.load("en_core_web_lg")


def preprocess(doc: str, **kwargs) -> str:
    """preprocessing a given document string. first all URLs are dropped,
    then stopwords are removed, then the resulting string is tokenised,
    lowercased, then lemmatised

    Parameters:
    -----------
    doc: str
        the string document to be processed
    kwargs: dict
        keyword arguments to be used with
        `gensim.parsing.preprocessing.simple_preprocess`, and with `spacy`'s
        nlp function

    Returns:
    --------
    out: str
        the processed string, all lowercased, no stop words included
    """
    # remove any hyperlinks
    doc_processed = re.sub(REGEX_URL, "", doc)

    # dropping stopwords
    doc_processed = remove_stopwords(doc_processed)

    # takes care of tokenising, lowercasing, and removing punctuations
    doc_processed = simple_preprocess(
        doc_processed,
        min_len=kwargs.get("min_len", 2),
        max_len=kwargs.get("max_len", 15),
    )

    # lemmatising the tokens
    doc_processed = nlp(" ".join(doc_processed), disable=kwargs.get("disable"))
    return " ".join([token.lemma_ for token in doc_processed])
