"""Typed, runtime-independent data contracts for the PoC."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal

from .text import count_words


QuestionType = Literal["recall", "understanding", "application", "other"]


@dataclass(frozen=True)
class Passage:
    text: str
    source_id: str = "source-passage"
    language: str | None = None

    @property
    def word_count(self) -> int:
        return count_words(self.text)

    @property
    def checksum(self) -> str:
        return sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QuizQuestion:
    question: str
    answer: str
    evidence_quote: str
    question_type: QuestionType = "other"
    rationale: str | None = None
    evidence_id: str | None = None


@dataclass(frozen=True)
class RunMetadata:
    model_id: str
    model_version: str
    runtime: str
    backend: str
    quantisation: str | None = None
    artifact_bytes: int | None = None
    inference_seconds: float | None = None
    peak_rss_bytes: int | None = None
    generated_tokens: int | None = None
    network_disabled_verified: bool = False
    artifact_sha256: str | None = None
    model_load_seconds: float | None = None
    first_token_seconds: float | None = None
    tokenizer_sha256: str | None = None


@dataclass(frozen=True)
class StudyAid:
    summary: str
    questions: tuple[QuizQuestion, ...]
    metadata: RunMetadata | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvidenceSpan:
    start: int
    end: int
    quote: str


@dataclass(frozen=True)
class CapsuleQuestion:
    id: str
    item: QuizQuestion
    evidence: EvidenceSpan
    answer_rubric: tuple[str, ...]


@dataclass(frozen=True)
class StudyPack:
    source_id: str
    source_sha256: str
    summary: str
    key_concepts: tuple[str, ...]
    questions: tuple[CapsuleQuestion, ...]
    metadata: RunMetadata | None


Outcome = Literal["correct", "partially_correct", "incorrect", "uncertain"]


@dataclass(frozen=True)
class PredictionAttempt:
    question_id: str
    text: str
    confidence: int | None
    rationale: str


@dataclass(frozen=True)
class FeedbackResult:
    outcome: Outcome
    rationale: str
    hint: str
    follow_up: str
    evidence: EvidenceSpan
    uncertainty: bool
