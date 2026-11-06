"""Read one passage and print its summary, quiz and run details."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .contracts import Passage, QuizQuestion
from .capsule import build_study_pack
from .learner_session import run_session
from .runtime_worker import LocalModel
from .study_aid import generate_study_aid
from .validation import normalise_for_match, validate_passage


def _check_offline() -> None:
    interfaces = Path("/sys/class/net")
    for _, name in socket.if_nameindex():
        if name != "lo" and int((interfaces / name / "flags").read_text(), 16) & 1:
            raise ValueError(f"Network interface {name} is up.")


def _practice_before_reveal(question: QuizQuestion) -> None:
    # Keep learner answers in this turn. The JSON output contains the study aid only.
    print(f"{question.question}\nPredict an answer before seeing the study aid.", file=sys.stderr)
    print("Your answer (Enter to skip): ", end="", file=sys.stderr, flush=True)
    prediction = sys.stdin.readline().strip()
    if not prediction:
        return
    print("Confidence (low, medium, high, or Enter to skip): ", end="", file=sys.stderr, flush=True)
    confidence = sys.stdin.readline().strip().lower()
    if confidence in {"low", "medium", "high"}:
        print(f"You reported {confidence} confidence.", file=sys.stderr)
    if normalise_for_match(prediction) != normalise_for_match(question.answer):
        print(f"Hint: the reference answer starts with {question.answer[0]!r}. Try once more.", file=sys.stderr)
        print("Your second answer (Enter to reveal): ", end="", file=sys.stderr, flush=True)
        sys.stdin.readline()
    print(f"Reference answer: {question.answer}\nSource: {question.evidence_quote}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("passage", type=Path, help="UTF-8 passage file")
    parser.add_argument("--model", type=Path, required=True, help="Local ExecuTorch model file")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Local matching tokenizer")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--quantisation", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=240, help="Seconds allowed for each model request")
    parser.add_argument("--prompt-template", type=Path, help="Local chat template with one {prompt} marker")
    parser.add_argument("--end-marker", help="Remove the model's trailing end-of-turn token from its response")
    parser.add_argument(
        "--source-recall", "--focused", dest="source_recall", action="store_true",
        help="Ask the model to choose recall terms and generate source-grounded cloze questions",
    )
    view = parser.add_mutually_exclusive_group()
    view.add_argument("--predict", action="store_true", help="Try a quiz question before revealing the study aid")
    view.add_argument("--capsule", action="store_true", help="Export the teacher-facing capsule with answer keys")
    view.add_argument("--session", action="store_true", help="Practice with prediction, reflection and feedback")
    parser.add_argument(
        "--kernel-library", action="append", type=Path, default=[],
        help="Load an extra native kernel library from the matched runtime build (repeatable)",
    )
    parser.add_argument(
        "--require-offline", action="store_true",
        help="Require non-loopback Linux interfaces to be down before and after generation",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.max_new_tokens <= 2048:
        parser.error("--max-new-tokens must be between 1 and 2048")
    if args.predict and not sys.stdin.isatty():
        parser.error("--predict requires an interactive terminal")

    try:
        from .executorch_runtime import ExecuTorchGenerator

        # Read the original bytes so answer offsets still match the submitted file.
        passage = Passage(args.passage.read_bytes().decode("utf-8"))
        validate_passage(passage)
        template = args.prompt_template.read_text(encoding="utf-8") if args.prompt_template else "{prompt}"
        generator = ExecuTorchGenerator(
            args.model, args.tokenizer, args.model_id, args.checkpoint,
            args.backend, args.quantisation, args.max_new_tokens, template,
            tuple(args.kernel_library), args.end_marker,
        )
        if args.require_offline:
            _check_offline()
        if args.session or args.capsule:
            print("Preparing the summary and three questions on this device…", file=sys.stderr, flush=True)
        with LocalModel(generator, args.timeout) as model:
            study_aid = generate_study_aid(passage, model, source_recall=args.source_recall)
            if args.require_offline and study_aid.metadata is not None:
                _check_offline()
                study_aid = replace(
                    study_aid, metadata=replace(study_aid.metadata, network_disabled_verified=True)
                )
            if args.capsule or args.session:
                pack = build_study_pack(passage, study_aid)
                if args.session:
                    session = run_session(
                        pack, model, sys.stdin.readline,
                        lambda message: print(message, file=sys.stderr, flush=True),
                        source_text=passage.text, source_recall=args.source_recall,
                    )
                    if args.require_offline:
                        _check_offline()
                    print(json.dumps({"events": session.events}, ensure_ascii=False, indent=2))
                    return 0
                print(json.dumps(asdict(pack), ensure_ascii=False, indent=2))
                return 0
    except KeyboardInterrupt:
        return 130
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"lwin: {error}", file=sys.stderr)
        return 1
    if args.predict:
        _practice_before_reveal(study_aid.questions[0])
    print(json.dumps(asdict(study_aid), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
