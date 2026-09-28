# utils/text_preprocessing.py

from pathlib import Path
import os
import tempfile
import re
import getpass

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


MODEL_TEXT_COLUMN = "lemmatized_text"

NGRAM_RANGE = (1, 2)
MIN_DF = 5
TEST_SIZE = 0.20
RANDOM_STATE = 42
LR_MAX_ITER = 1000



_NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


@st.cache_resource
def ensure_nltk_data():

    user_identifier = getattr(
        os,
        "getuid",
        lambda: getpass.getuser()
    )()

    nltk_dir = (
        Path(tempfile.gettempdir())
        / f"nltk_data_{user_identifier}"
    )

    nltk_dir.mkdir(
        mode=0o700,
        parents=True,
        exist_ok=True
    )

    try:
        nltk_dir.chmod(0o700)
    except OSError:
        pass

    # Put our private directory FIRST
    if str(nltk_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_dir))

    for package, resource_path in _NLTK_RESOURCES.items():

        try:
            nltk.data.find(resource_path)

        except LookupError:

            success = nltk.download(
                package,
                download_dir=str(nltk_dir),
                quiet=True
            )

            if not success:
                raise RuntimeError(
                    f"Could not download NLTK resource: {package}"
                )

    return str(nltk_dir)


ensure_nltk_data()


KEEP_WORDS = {
    "no",
    "not",
    "nor",
    "never",
    "but",
    "very",
    "too",
    "only",
    "against",
}


ENGLISH_STOPWORDS = (
    set(stopwords.words("english")) - KEEP_WORDS
)


LEMMATIZER = WordNetLemmatizer()


CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "mustn't": "must not",
    "mightn't": "might not",
}



def strip_html(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Expand contractions
    for contraction, replacement in CONTRACTIONS.items():
        text = text.replace(contraction, replacement)

    # Remove HTML
    text = strip_html(text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Keep alphabetic characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def lemmatize_text(text):

    words = text.split()

    words = [
        word
        for word in words
        if word not in ENGLISH_STOPWORDS
    ]

    words = [
        LEMMATIZER.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


def preprocess(text):

    text = clean_text(text)

    text = lemmatize_text(text)

    return text