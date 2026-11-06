"""Count words and identify source sentences without changing their text."""

import re
from unicodedata import category


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|(?<=[।॥。！？])\s*|\r?\n[ \t]*\r?\n")


def source_sentences(text: str) -> dict[str, str]:
    """Assign stable IDs to source spans while preserving the passage text."""
    sentences = (part.strip() for part in _SENTENCE_BREAK.split(text))
    return {f"s{index}": sentence for index, sentence in enumerate(filter(None, sentences), 1)}


def source_offsets(text: str) -> dict[str, tuple[int, int]]:
    offsets = {}
    cursor = 0
    for source_id, sentence in source_sentences(text).items():
        start = text.index(sentence, cursor)
        cursor = start + len(sentence)
        offsets[source_id] = (start, cursor)
    return offsets


def model_source(text: str) -> str:
    """Keep embedded chat delimiters from opening a new role in the runner template."""
    return text.replace("<|", "< |")


def is_word_character(character: str) -> bool:
    return category(character)[0] in "LMN" or character in "\u200c\u200d"


def count_words(text: str) -> int:
    """Count runs of letters and numbers, keeping combining marks and joiners."""
    count = 0
    in_word = False
    for index, character in enumerate(text):
        if character.isalnum():
            if not in_word:
                count += 1
            in_word = True
        elif in_word and character in "'’‐‑-." and index + 1 < len(text):
            following = text[index + 1]
            in_word = following.isalnum()
            if character == ".":
                in_word = in_word and text[index - 1].isdecimal() and following.isdecimal()
        elif not is_word_character(character):
            in_word = False
    return count
