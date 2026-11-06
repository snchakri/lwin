"""Run a local ExecuTorch model and measure the generation session."""

from __future__ import annotations

import ctypes
import hashlib
import resource
from dataclasses import dataclass, replace
from functools import cached_property
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from .contracts import RunMetadata


@dataclass(frozen=True)
class ExecuTorchGenerator:
    model: Path
    tokenizer: Path
    model_id: str
    checkpoint: str
    backend: str
    quantisation: str
    max_new_tokens: int
    prompt_template: str = "{prompt}"
    kernel_libraries: tuple[Path, ...] = ()
    end_marker: str | None = None

    @cached_property
    def _asset_metadata(self) -> RunMetadata:
        digest = hashlib.sha256()
        artifact_bytes = 0
        with self.model.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
                artifact_bytes += len(chunk)
        if not artifact_bytes:
            raise ValueError("Model artifact is empty.")
        tokenizer = self.tokenizer.read_bytes()
        if not tokenizer:
            raise ValueError("Tokenizer must be a nonempty local file.")
        if self.prompt_template.count("{prompt}") != 1:
            raise ValueError("Prompt template must contain exactly one {prompt} marker.")

        return RunMetadata(
            model_id=self.model_id,
            model_version=self.checkpoint,
            runtime=f"ExecuTorch {version('executorch')}",
            backend=self.backend,
            quantisation=self.quantisation,
            artifact_bytes=artifact_bytes,
            artifact_sha256=digest.hexdigest(),
            tokenizer_sha256=hashlib.sha256(tokenizer).hexdigest(),
        )

    @cached_property
    def _runtime(self):
        # Wait until a passage passes intake before importing or loading the native runtime.
        from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner

        for library in self.kernel_libraries:
            ctypes.CDLL(str(library.resolve(strict=True)))
        return TextLLMRunner(str(self.model), str(self.tokenizer)), GenerationConfig

    def generate(
        self, prompt: str, *, max_new_tokens: int | None = None, response_prefix: str = ""
    ) -> tuple[str, RunMetadata]:
        metadata = self._asset_metadata
        tokens: list[str] = []
        generated_tokens: int | None = None
        model_load_seconds: float | None = None
        first_token_seconds: float | None = None

        def collect_stats(stats) -> None:
            nonlocal generated_tokens, model_load_seconds, first_token_seconds
            generated_tokens = stats.num_generated_tokens
            scale = stats.SCALING_FACTOR_UNITS_PER_SECOND
            if stats.model_load_start_ms and stats.model_load_end_ms >= stats.model_load_start_ms:
                model_load_seconds = (stats.model_load_end_ms - stats.model_load_start_ms) / scale
            if stats.inference_start_ms and stats.first_token_ms >= stats.inference_start_ms:
                first_token_seconds = (stats.first_token_ms - stats.inference_start_ms) / scale

        start = perf_counter()
        runner, config_type = self._runtime
        # Clear the previous prompt while keeping the same model loaded for the next choice.
        runner.reset()
        limit = min(self.max_new_tokens, max_new_tokens) if max_new_tokens is not None else self.max_new_tokens
        config = config_type(echo=False, temperature=0.0, max_new_tokens=limit)
        runner.generate(
            self.prompt_template.replace("{prompt}", prompt) + response_prefix,
            config, tokens.append, collect_stats,
        )
        response = "".join(tokens)
        if self.end_marker:
            response = response.rstrip().removesuffix(self.end_marker)
        return response_prefix + response, replace(
            metadata,
            inference_seconds=perf_counter() - start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            generated_tokens=generated_tokens,
            model_load_seconds=model_load_seconds,
            first_token_seconds=first_token_seconds,
        )
