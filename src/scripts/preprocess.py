import regex as re
from typing import List


def clean_text(text: str) -> str:
    """
    Cleans text from punctuation, URLs, special characters, multiple spaces and lowercases.
    Args:
        text (str): The string to clean.
    Returns:
        str: The cleaned string.
    """

    no_urls = re.sub(r"http\S+", "", text)
    no_special_ch = re.sub(r"([^A-Za-z])|(\w+:\/\/\S+)", " ", no_urls)
    lowercased_str = no_special_ch.lower()
    cleaned_text = " ".join(lowercased_str.split())

    return cleaned_text


def collect_lemmas(text: str, nlp) -> List[str]:
    """
    Lemmatizes text using spaCy pipeline.
    Args:
        text (str): A text to extract lemmas from.
        nlp: A spaCy pipeline.
    Returns:
        List[str]: A list with lemmas.
    """

    lemmas = []

    doc = nlp(text)

    for token in doc:
        lemmas.append(token.lemma_)

    return lemmas


def collect_nn_adj(text: str, nlp, pos_tags: List[str]) -> List[str]:
    """
    Collects lemmas only with a specified POS tag.
    Args:
        text (str): A text to extract lemmas from.
        nlp: A spaCy pipeline.
        pos_tags (List[str]): A list with POS tags.
    Returns:
        List[str]: A list with lemmas.
    """

    nn_adj = []

    doc = nlp(text)

    for token in doc:
        if token.pos_ in pos_tags:
            nn_adj.append(token.lemma_)
    return nn_adj


def rm_stops(text: List[str], stopwords: List[str]) -> List[str]:
    """
    Removes stopwords from tokenized/lemmatized text.
    Args:
        text (List[str]): A list with tokens/lemmas.
        stopwords (List[str]): A list of stopwords.
    Returns:
       List[str]: A list with tokens/lemmas without stopwords.
    """

    no_stopwords = []

    for word in text:
        if word not in stopwords:
            no_stopwords.append(word)

    return no_stopwords
