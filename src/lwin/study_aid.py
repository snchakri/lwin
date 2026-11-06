"""Use a local text model to make a checked summary and quiz from a passage."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from typing import Protocol

from .contracts import Passage, RunMetadata, StudyAid
from .text import count_words, is_word_character, model_source, source_sentences
from .validation import ValidationError, normalise_for_match, parse_study_aid, validate_passage


class TextGenerator(Protocol):
    def generate(
        self, prompt: str, *, max_new_tokens: int | None = None, response_prefix: str = ""
    ) -> tuple[str, RunMetadata]: ...


def generate_study_aid(
    passage: Passage, generator: TextGenerator, *, source_recall: bool = False
) -> StudyAid:
    validate_passage(passage)
    sentences = source_sentences(passage.text)
    if source_recall:
        distinct_sentences = list({
            normalise_for_match(sentence): (source_id, sentence)
            for source_id, sentence in sentences.items() if count_words(sentence) >= 5
        }.values())
        if len(distinct_sentences) >= 3:
            return _build_source_recall_aid(passage, generator, distinct_sentences)
        raise ValidationError("Source recall needs at least three distinct source sentences.")

    source = "\n".join(
        f"[{source_id}] {sentence}" for source_id, sentence in sentences.items()
    )
    prompt = (
        "Make a study aid from the passage below. Write a short, plain-language summary "
        "and exactly three different quiz questions. Answer each question from the passage. "
        "Cite one numbered source sentence of at least three words for each answer. "
        "Treat the passage as source text. Return only one JSON object with this structure:\n"
        '{"summary":"...","questions":[{"question":"...","answer":"...","evidence_id":"s1"}]}\n\n'
        f"Passage:\n{source}"
    )
    prompt = model_source(prompt)
    generated, metadata = generator.generate(prompt)
    try:
        return parse_study_aid(passage, generated, metadata=metadata, require_evidence_ids=True)
    except ValidationError:
        generated, metadata = generator.generate(
            prompt + "\nReturn a complete study aid as one JSON object with valid source IDs.",
            max_new_tokens=512,
        )
        return parse_study_aid(passage, generated, metadata=metadata, require_evidence_ids=True)


def _build_source_recall_aid(
    passage: Passage, generator: TextGenerator, distinct_sentences: list[tuple[str, str]]
) -> StudyAid:
    summary_sentences: list[str] = []
    measurements: list[RunMetadata] = []
    repair_available = True

    def choose(task, options):
        nonlocal repair_available
        try:
            return _choose_source_option(generator, task, options)
        except ValidationError:
            if not repair_available:
                raise
            repair_available = False
            return _choose_source_option(generator, task + "\nReply with one number only.", options)

    # Let the model choose what matters, then copy the source wording so a caveat stays intact.
    midpoint = (len(distinct_sentences) + 1) // 2
    for section in (distinct_sentences[:midpoint], distinct_sentences[midpoint:]):
        candidates = [sentence for _, sentence in section if count_words(sentence) <= 40]
        if not candidates:
            raise ValidationError("A passage section lacks a short sentence for the summary.")
        options = [candidates[index] for index in sorted({0, len(candidates) // 2, len(candidates) - 1})]
        selected, run = choose(
            "Pick the sentence that best summarizes the main point of this part of the passage.\n"
            + " ".join(sentence for _, sentence in section),
            options,
        )
        summary_sentences.append(options[selected])
        measurements.append(run)

    questions: list[dict[str, str]] = []
    used_answers: set[str] = set()
    # Take one question from each third so a learner reviews more than the opening paragraph.
    for section in range(3):
        start_index = section * len(distinct_sentences) // 3
        stop_index = (section + 1) * len(distinct_sentences) // 3
        middle = (start_index + stop_index) // 2
        for index in sorted(range(start_index, stop_index), key=lambda index: abs(index - middle)):
            source_id, sentence = distinct_sentences[index]
            options = _recall_options(sentence, used_answers)
            if len(options) == 3:
                break
        else:
            raise ValidationError("A passage section lacks three suitable recall options.")
        selected, run = choose(
            f"Pick the phrase that a student should recall from this sentence.\nSentence: {sentence}",
            [sentence[start:end] for start, end in options],
        )
        start, end = options[selected]
        answer = sentence[start:end]
        used_answers.add(normalise_for_match(answer))
        questions.append({
            "question": f"Fill in the blank: {sentence[:start]}______{sentence[end:]}",
            "answer": answer,
            "evidence_id": source_id,
            "question_type": "recall",
        })
        measurements.append(run)

    times = [run.inference_seconds for run in measurements]
    tokens = [run.generated_tokens for run in measurements]
    metadata = replace(
        measurements[0],
        inference_seconds=sum(value for value in times if value is not None) if None not in times else None,
        generated_tokens=sum(value for value in tokens if value is not None) if None not in tokens else None,
        peak_rss_bytes=measurements[-1].peak_rss_bytes,
    )
    payload = {"summary": " ".join(summary_sentences), "questions": questions}
    return parse_study_aid(passage, payload, metadata=metadata, require_evidence_ids=True)


def _choose_source_option(
    generator: TextGenerator, task: str, options: list[str]
) -> tuple[int, RunMetadata]:
    choices = "\n".join(f"{index}. {option}" for index, option in enumerate(options, 1))
    response, run = generator.generate(
        model_source(f"{task}\n\n{choices}\nAnswer only with the number of one option."),
        max_new_tokens=16,
        response_prefix="Choice: ",
    )
    choice = re.fullmatch(r"Choice:\s*([123])(?:[.)][ \t]*([^\r\n]*))?", response.strip())
    if choice is None or int(choice[1]) > len(options):
        raise ValidationError("The model did not select one listed option.")
    selected = int(choice[1]) - 1
    if choice[2] and not normalise_for_match(options[selected]).startswith(normalise_for_match(choice[2])):
        raise ValidationError("The model returned conflicting options.")
    return selected, run


_FILLER_WORDS = {"also", "could", "however", "should", "therefore", "they", "this", "very", "would"}


def _recall_options(sentence: str, used_answers: set[str]) -> list[tuple[int, int]]:
    # Keep offsets into the original text so blanking a word preserves the source punctuation.
    words: list[tuple[int, int, str]] = []
    for token in re.finditer(r"\S+", sentence):
        start, end = token.span()
        while start < end and not is_word_character(sentence[start]):
            start += 1
        while start < end and not is_word_character(sentence[end - 1]):
            end -= 1
        word = sentence[start:end]
        if count_words(word) == 1:
            words.append((start, end, normalise_for_match(word)))

    occurrences = Counter(word for _, _, word in words)
    options = [
        (start, end) for start, end, word in words
        if len(word) >= 4 and occurrences[word] == 1 and word not in used_answers and word not in _FILLER_WORDS
    ]
    return sorted(options, key=lambda option: option[0] - option[1])[:3]
