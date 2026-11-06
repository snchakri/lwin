"""Deterministic input/output validation. These checks do not prove semantic correctness."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import get_args

from .contracts import Passage, QuestionType, QuizQuestion, RunMetadata, StudyAid
from .text import count_words, is_word_character, source_sentences


class ValidationError(ValueError):
    """Raised when an input or generated study aid violates the PoC contract."""


_TYPOGRAPHY = str.maketrans("‘’“”‐‑‒–—", "''\"\"-----")


def normalise_for_match(value: str, *, compatibility: bool = False) -> str:
    """Fold case and common typography while preserving meaningful characters."""
    value = unicodedata.normalize("NFKC" if compatibility else "NFC", value)
    value = value.casefold().translate(_TYPOGRAPHY)
    value = re.sub(r"\s*-{1,2}\s*", "-", value)
    return " ".join(value.split())


def _has_source_span(source: str, quote: str) -> bool:
    start = source.find(quote)
    while start != -1:
        end = start + len(quote)
        joins_previous_word = start > 0 and is_word_character(quote[0]) and is_word_character(source[start - 1])
        joins_next_word = end < len(source) and is_word_character(quote[-1]) and is_word_character(source[end])
        if not (joins_previous_word or joins_next_word):
            return True
        start = source.find(quote, start + 1)
    return False


def _require_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a non-blank string.")


def _validate_run_metadata(metadata: RunMetadata) -> None:
    for field in ("model_id", "model_version", "runtime", "backend"):
        _require_text(getattr(metadata, field), f"Run metadata {field}")
    if metadata.quantisation is not None:
        _require_text(metadata.quantisation, "Run metadata quantisation")
    for field in ("artifact_bytes", "peak_rss_bytes", "generated_tokens"):
        value = getattr(metadata, field)
        if value is not None and (type(value) is not int or value < 0):
            raise ValidationError(f"Run metadata {field} must be a non-negative integer.")
    for field in ("inference_seconds", "model_load_seconds", "first_token_seconds"):
        seconds = getattr(metadata, field)
        if seconds is not None and (
            type(seconds) not in (int, float) or seconds < 0
            or (isinstance(seconds, float) and not math.isfinite(seconds))
        ):
            raise ValidationError(f"Run metadata {field} must be finite and non-negative.")
    if type(metadata.network_disabled_verified) is not bool:
        raise ValidationError("Run metadata network_disabled_verified must be a boolean.")
    for field in ("artifact_sha256", "tokenizer_sha256"):
        sha = getattr(metadata, field)
        if sha is not None and (not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha)):
            raise ValidationError(f"Run metadata {field} must be a hexadecimal SHA-256 digest.")


def validate_passage(
    passage: Passage,
    *,
    min_words: int = 150,
    max_words: int = 300,
    enforce_range: bool = True,
) -> None:
    """Validate input before generation using Unicode-aware word counts."""
    if not isinstance(passage, Passage):
        raise ValidationError("Expected a Passage.")
    _require_text(passage.text, "Passage text")
    _require_text(passage.source_id, "Passage source ID")
    if passage.language is not None:
        _require_text(passage.language, "Passage language")
    if type(min_words) is not int or type(max_words) is not int or not 1 <= min_words <= max_words:
        raise ValidationError("Passage word limits must be positive integers in ascending order.")
    if type(enforce_range) is not bool:
        raise ValidationError("enforce_range must be a boolean.")
    if enforce_range:
        count = passage.word_count
        if not min_words <= count <= max_words:
            raise ValidationError(
                f"Passage contains {count} words; expected approximately "
                f"{min_words}–{max_words}."
            )


def validate_study_aid(
    passage: Passage,
    study_aid: StudyAid,
    *,
    min_questions: int = 3,
) -> None:
    """Validate the output against a passage whose word range was checked at intake."""
    validate_passage(passage, enforce_range=False)
    if not isinstance(study_aid, StudyAid):
        raise ValidationError("Expected a StudyAid.")
    _require_text(study_aid.summary, "Summary")
    if not 3 <= count_words(study_aid.summary) <= 80:
        raise ValidationError("Summary must contain 3–80 words.")
    if not isinstance(study_aid.questions, tuple):
        raise ValidationError("StudyAid questions must be a tuple.")
    if study_aid.metadata is not None:
        if not isinstance(study_aid.metadata, RunMetadata):
            raise ValidationError("StudyAid metadata must be RunMetadata or None.")
        _validate_run_metadata(study_aid.metadata)
    if not isinstance(study_aid.warnings, tuple):
        raise ValidationError("StudyAid warnings must be a tuple.")
    for warning in study_aid.warnings:
        _require_text(warning, "StudyAid warning")

    if type(min_questions) is not int or min_questions < 1:
        raise ValidationError("min_questions must be a positive integer.")
    if len(study_aid.questions) < min_questions:
        raise ValidationError(
            f"At least {min_questions} questions are required; got {len(study_aid.questions)}."
        )

    seen: set[str] = set()
    source = normalise_for_match(passage.text)
    sentences = source_sentences(passage.text)
    for index, question in enumerate(study_aid.questions, start=1):
        if not isinstance(question, QuizQuestion):
            raise ValidationError(f"Question {index} must be a QuizQuestion.")
        _require_text(question.question, f"Question {index} text")
        _require_text(question.answer, f"Question {index} answer")
        _require_text(question.evidence_quote, f"Question {index} evidence quote")
        if question.question_type not in get_args(QuestionType):
            raise ValidationError(f"Question {index} has an invalid question type.")
        if question.rationale is not None:
            _require_text(question.rationale, f"Question {index} rationale")

        key = normalise_for_match(question.question, compatibility=True).rstrip(" .?!؟…。！？।")
        if key in seen:
            raise ValidationError(f"Duplicate question detected at index {index}.")
        seen.add(key)

        quote = normalise_for_match(question.evidence_quote)
        if count_words(quote) < 3:
            raise ValidationError(f"Evidence quote for question {index} must contain at least three words.")
        if question.evidence_id is not None:
            _require_text(question.evidence_id, f"Question {index} evidence ID")
            if sentences.get(question.evidence_id) != question.evidence_quote:
                raise ValidationError(f"Evidence ID for question {index} does not identify its source quote.")
        elif not _has_source_span(source, quote):
            raise ValidationError(
                f"Evidence quote for question {index} does not occur in the supplied passage."
            )


def _unique_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    obj: dict[str, object] = {}
    for key, value in pairs:
        if key in obj:
            raise ValidationError("Generated JSON contains a duplicate field.")
        obj[key] = value
    return obj


def parse_study_aid(
    passage: Passage,
    payload: object,
    *,
    metadata: RunMetadata | None = None,
    require_evidence_ids: bool = False,
) -> StudyAid:
    """Validate a JSON object or fenced JSON response and attach local metadata."""
    if isinstance(payload, str):
        raw = payload.strip()
        fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", raw, re.IGNORECASE | re.DOTALL)
        try:
            payload = json.loads(fence[1] if fence else raw, object_pairs_hook=_unique_json_keys)
        except ValidationError:
            raise
        except (ValueError, RecursionError) as error:
            raise ValidationError("The model response was not a complete JSON study aid.") from error
    if not isinstance(payload, dict) or not {"summary", "questions"} <= payload.keys():
        raise ValidationError("Generated output must be an object with summary and questions fields.")
    raw_questions = payload["questions"]
    if not isinstance(raw_questions, list):
        raise ValidationError("Generated questions must be an array.")

    validate_passage(passage, enforce_range=False)
    sentences = source_sentences(passage.text)
    required = {"question", "answer"}
    allowed = required | {"evidence_id", "evidence_quote", "question_type", "rationale"}
    questions: list[QuizQuestion] = []
    for index, item in enumerate(raw_questions, start=1):
        if not isinstance(item, dict) or not required <= item.keys() <= allowed:
            raise ValidationError(
                f"Question {index} must contain question, answer and a source reference."
            )
        fields = dict(item)
        evidence_id = fields.get("evidence_id")
        if require_evidence_ids or evidence_id is not None:
            if not isinstance(evidence_id, str) or evidence_id not in sentences:
                raise ValidationError(f"Question {index} must cite a valid evidence ID.")
            fields["evidence_quote"] = sentences[evidence_id]
        elif "evidence_quote" not in fields:
            raise ValidationError(f"Question {index} must contain an evidence quote.")
        rationale = fields.get("rationale")
        if isinstance(rationale, str) and not rationale.strip():
            fields["rationale"] = None
        questions.append(QuizQuestion(**fields))

    study_aid = StudyAid(payload["summary"], tuple(questions), metadata=metadata)
    validate_study_aid(passage, study_aid)
    return study_aid
