"""
Query decomposition using spaCy.
"""

import spacy

nlp = spacy.load("en_core_web_sm")


def decompose_query(query):
    """
    Returns:
    {
        "full_query": str,
        "sub_queries": list[str]
    }
    """

    doc = nlp(query)

    aspects = []

    noun_chunks = list(doc.noun_chunks)

    for chunk in noun_chunks:
        phrase = chunk.text.strip()

        if len(phrase) > 2:
            aspects.append(phrase)

    for token in doc:
        if token.pos_ == "VERB":
            aspects.append(token.lemma_)

    aspects = list(dict.fromkeys(aspects))

    return {
        "full_query": query,
        "sub_queries": aspects if aspects else [query]
    }