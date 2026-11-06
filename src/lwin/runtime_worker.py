"""Keep one native model in a process that can be stopped if it stalls."""

from __future__ import annotations

import math
import multiprocessing as mp
import os


class ModelError(RuntimeError):
    pass


class ModelTimeout(ModelError):
    pass


_MAX_TEXT = 65_536


def _run_model_worker(generator, channel, parent_channel) -> None:
    parent_channel.close()
    # The runner can echo tokens itself. Keep those echoes out of the caller's JSON.
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
    with channel:
        while True:
            try:
                prompt, limit, prefix = channel.recv()
            except EOFError:
                return
            try:
                response, metadata = generator.generate(prompt, max_new_tokens=limit, response_prefix=prefix)
                if len(response) > _MAX_TEXT:
                    raise ModelError("Model output exceeded the text budget.")
                channel.send(("ok", (response, metadata)))
            except Exception as error:
                channel.send(("error", str(error)[:1000]))
                return


class LocalModel:
    def __init__(self, generator, timeout: float = 240):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Model timeout must be finite and positive.")
        self.generator = generator
        self.timeout = timeout
        self._process = None
        self._channel = None
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def generate(self, prompt: str, *, max_new_tokens: int | None = None, response_prefix: str = ""):
        if self._closed:
            raise ModelError("The model session is closed.")
        if len(prompt) > 12_000 or len(response_prefix) > 128:
            raise ModelError("Model input exceeded the text budget.")
        if max_new_tokens is not None and not 1 <= max_new_tokens <= 2048:
            raise ModelError("Generation limit must be between 1 and 2048 tokens.")
        if self._process is None:
            context = mp.get_context("fork")
            parent, child = context.Pipe()
            process = context.Process(target=_run_model_worker, args=(self.generator, child, parent), daemon=True)
            process.start()
            child.close()
            self._process, self._channel = process, parent
        try:
            self._channel.send((prompt, max_new_tokens, response_prefix))
            if not self._channel.poll(self.timeout):
                raise ModelTimeout(f"Local model exceeded the {self.timeout:g} s request timeout.")
            status, result = self._channel.recv()
            if status != "ok":
                raise ModelError(result)
            return result
        except (EOFError, BrokenPipeError, ConnectionResetError) as error:
            self.close()
            raise ModelError("Local model worker exited before returning a result.") from error
        except BaseException:
            self.close()
            raise

    def close(self):
        self._closed = True
        if self._process is not None:
            self._channel.close()
            self._process.join(0.25)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(1)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
            self._process.close()
            self._process = None
            self._channel = None
