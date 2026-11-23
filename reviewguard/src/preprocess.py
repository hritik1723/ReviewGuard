"""Text preprocessing pipeline for ReviewGuard.

Classical NLP steps:
- Lowercase normalization
- URL / HTML tag removal
- Punctuation & special character stripping
- Tokenization
- Stopword filtering
- WordNet lemmatization
"""

import re
from typing import List, Union
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Compile regex patterns for efficiency
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
PUNCT_PATTERN = re.compile(r"[^\w\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")

# Initialize stop words and lemmatizer safely
try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

try:
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    nltk.download("wordnet", quiet=True)
    LEMMATIZER = WordNetLemmatizer()


def clean_text_basic(text: str) -> str:
    """Perform fast URL removal, HTML removal, and whitespace normalization."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    return text.strip()


def tokenize_and_lemmatize(
    text: str,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
) -> List[str]:
    """Clean text, tokenize on whitespace/regex, filter stopwords, and lemmatize."""
    text = clean_text_basic(text).lower()
    text = PUNCT_PATTERN.sub(" ", text)
    tokens = text.split()

    cleaned_tokens = []
    for token in tokens:
        if not token.isalpha() or len(token) <= 1:
            continue
        if remove_stopwords and token in STOP_WORDS:
            continue
        if lemmatize:
            token = LEMMATIZER.lemmatize(token)
        cleaned_tokens.append(token)

    return cleaned_tokens


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
) -> str:
    """Clean and lemmatize raw review string into a normalized space-separated string."""
    tokens = tokenize_and_lemmatize(
        text, remove_stopwords=remove_stopwords, lemmatize=lemmatize
    )
    return " ".join(tokens)


def preprocess_series(series, remove_stopwords: bool = True, lemmatize: bool = True):
    """Batch preprocess a pandas Series of text."""
    return series.astype(str).map(
        lambda t: preprocess_text(
            t, remove_stopwords=remove_stopwords, lemmatize=lemmatize
        )
    )
