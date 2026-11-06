"""Build the auditable study pack from a validated study aid."""

from .contracts import CapsuleQuestion, EvidenceSpan, Passage, StudyAid, StudyPack
from .text import source_offsets
from .validation import ValidationError, normalise_for_match, validate_study_aid


def build_study_pack(passage: Passage, aid: StudyAid) -> StudyPack:
    validate_study_aid(passage, aid)
    offsets = source_offsets(passage.text)
    concepts = []
    used_spans = set()
    questions = []
    for index, question in enumerate(aid.questions, 1):
        span = offsets.get(question.evidence_id)
        if span is None:
            start = passage.text.find(question.evidence_quote)
            span = (start, start + len(question.evidence_quote))
        start, end = span
        if start < 0 or passage.text[start:end] != question.evidence_quote:
            raise ValidationError(f"Question {index} does not have an exact source span.")
        key = normalise_for_match(question.evidence_quote)
        if key in used_spans:
            raise ValidationError(f"Question {index} repeats a source excerpt.")
        used_spans.add(key)
        if question.answer.casefold() not in {term.casefold() for term in concepts}:
            concepts.append(question.answer)
        questions.append(CapsuleQuestion(
            f"q{index}", question, EvidenceSpan(start, end, question.evidence_quote),
            (f"Identify {question.answer} or a source-supported equivalent.",),
        ))
    return StudyPack(passage.source_id, passage.checksum, aid.summary, tuple(concepts[:5]), tuple(questions), aid.metadata)
