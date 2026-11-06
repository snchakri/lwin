"""Control what a learner can see and record only temporary observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from unicodedata import category
from uuid import uuid4

from .contracts import FeedbackResult, Outcome, PredictionAttempt, StudyPack
from .tutor import assess, feedback_for


@dataclass
class SessionState:
    pack: StudyPack
    session_id: str = field(default_factory=lambda: uuid4().hex)
    phase: str = "prediction"
    prediction: PredictionAttempt | None = None
    reflection: str = ""
    answer: str = ""
    feedback: FeedbackResult | None = None
    follow_up: str = ""
    events: list[dict] = field(default_factory=list)

    def _transition(self, expected: str, next_phase: str, **facts) -> None:
        if self.phase != expected:
            raise ValueError(f"Expected {expected} phase, found {self.phase}.")
        self.phase = next_phase
        self.events.append({
            "session_id": self.session_id,
            "phase": next_phase,
            "at": datetime.now(timezone.utc).isoformat(),
            **facts,
        })

    def submit_prediction(self, text: str, confidence: int | None, rationale: str) -> None:
        if confidence is not None and (type(confidence) is not int or not 0 <= confidence <= 100):
            raise ValueError("Confidence must be an integer between 0 and 100.")
        if not text.strip() and (confidence is not None or rationale.strip()):
            raise ValueError("Skipped predictions have no confidence or rationale.")
        question = self.pack.questions[0]
        self._transition("prediction", "reveal", question_id=question.id, skipped=not bool(text.strip()),
                         confidence=confidence)
        self.prediction = PredictionAttempt(question.id, text, confidence, rationale)

    def reveal_prediction(self, alignment: str):
        if alignment not in {"aligned", "partially_aligned", "not_aligned", "cannot_assess"}:
            raise ValueError("Invalid prediction comparison.")
        question = self.pack.questions[0]
        self._transition("reveal", "reflection", question_id=question.id,
                         evidence_id=question.item.evidence_id, alignment=alignment)
        return question

    def complete_reflection(self, text: str) -> None:
        self._transition("reflection", "answer", reflected=bool(text.strip()))
        self.reflection = text

    def submit_answer(self, text: str) -> None:
        self._transition("answer", "assessment", question_id=self.pack.questions[1].id)
        self.answer = text

    def show_feedback(self, feedback: FeedbackResult) -> None:
        self._transition("assessment", "follow_up", outcome=feedback.outcome)
        self.feedback = feedback

    def finish(
        self, text: str, *, reveal: bool, retry_outcome: Outcome | None = None, self_study_revealed: bool = False
    ) -> None:
        self._transition("follow_up", "complete", retried=bool(text.strip()), answer_revealed=reveal,
                         retry_outcome=retry_outcome, self_study_revealed=self_study_revealed)
        self.follow_up = text


def run_session(pack: StudyPack, model, read, write, *, source_text: str = "", source_recall: bool = False) -> SessionState:
    def show(text):
        # Passage text goes to a terminal, so drop control characters before printing it.
        write("".join(char for char in text if char in "\n\t" or category(char) not in {"Cc", "Cf"}))

    def ask(prompt):
        show(prompt)
        return read().strip()[:800]

    state = SessionState(pack)
    prediction_question, tutor_question, self_study_question = pack.questions[:3]
    if source_text:
        show(f"Read the passage before starting.\n\n{source_text}")
    if source_recall:
        show("For each blank, try to recall the word used in the passage.")
    show("\nQuestion 1 of 3")
    show(prediction_question.item.question)
    prediction = ask("Predict an answer, or press Enter to skip.")
    confidence = None
    rationale = ""
    if prediction:
        confidence_text = ask("Optional confidence from 0 to 100. Enter skips.")
        if confidence_text and confidence_text.isascii() and confidence_text.isdecimal() and len(confidence_text) <= 3:
            value = int(confidence_text)
            confidence = value if 0 <= value <= 100 else None
        if confidence_text and confidence is None:
            show("Confidence skipped. Enter a whole number from 0 to 100 to report it next time.")
        rationale = ask("Optional reason for your prediction. Enter skips.")
    state.submit_prediction(prediction, confidence, rationale)
    outcome, _ = assess(model, prediction_question, prediction, source_recall=source_recall)
    alignment = {
        "correct": "aligned", "partially_correct": "partially_aligned",
        "incorrect": "not_aligned", "uncertain": "cannot_assess",
    }[outcome]
    revealed = state.reveal_prediction(alignment)
    show(f"Reference answer: {revealed.item.answer}\nSource: {revealed.evidence.quote}")
    if prediction:
        comparison = feedback_for(revealed, outcome, source_recall=source_recall)
        show(f"Check your prediction. {comparison.rationale}")
    state.complete_reflection(ask("Which detail in the source confirmed or changed your prediction? Enter skips."))
    show("\nQuestion 2 of 3")
    show(tutor_question.item.question)
    state.submit_answer(ask("Your answer. Enter skips."))
    outcome, _ = assess(model, tutor_question, state.answer, source_recall=source_recall)
    feedback = feedback_for(tutor_question, outcome, source_recall=source_recall)
    state.show_feedback(feedback)
    show(f"Feedback: {feedback.rationale}")
    if feedback.hint:
        show(feedback.hint)
    follow_up = ask(f"{feedback.follow_up} Enter skips.")
    retry_outcome = None
    if follow_up and outcome != "correct":
        retry_outcome, _ = assess(model, tutor_question, follow_up, source_recall=source_recall)
        show(f"Follow-up: {feedback_for(tutor_question, retry_outcome, source_recall=source_recall).rationale}")
    reveal = ask("Show the reference answer and its source? Type yes to reveal.").lower() == "yes"
    if reveal:
        show(f"Reference answer: {tutor_question.item.answer}\nSource: {feedback.evidence.quote}")
    show("\nQuestion 3 of 3 · Self-study")
    show(self_study_question.item.question)
    self_study_revealed = ask("Think of an answer, then type yes to check it. Enter skips.").lower() == "yes"
    if self_study_revealed:
        show(f"Reference answer: {self_study_question.item.answer}\nSource: {self_study_question.evidence.quote}")
    # The summary may include an answer, so show it after all three questions.
    show(f"\nSummary: {pack.summary}")
    state.finish(follow_up, reveal=reveal, retry_outcome=retry_outcome, self_study_revealed=self_study_revealed)
    return state
