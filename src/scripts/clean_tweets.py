import regex as re


def clean_text(doc: str) -> str:
    cleaned = re.sub(r"http\S+", "", doc)
    cleaned = cleaned.replace("\n", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned
