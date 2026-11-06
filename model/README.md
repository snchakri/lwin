# Local study model

The bundle uses `Qwen/Qwen2.5-0.5B-Instruct` at revision
`7ae557604adf67be50417f59c2c2f167def9a775`. The local `.pte` is the third-party
XNNPACK export at revision `cff4d0471726bcb582f0d013a74ba36f26d86ff6`, with
8-bit dynamic activations, 4-bit GPTQ weights and a 2,048-token context.
The model weights are licensed under Apache 2.0. [Checkpoint](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775) · [Export](https://huggingface.co/experimentalmachines/Qwen2.5-0.5B-Instruct-ExecuTorch/tree/cff4d0471726bcb582f0d013a74ba36f26d86ff6)

| Bundle file | SHA-256 |
| --- | --- |
| `study-model.pte` (417,495,296 bytes) | `b20eeac40737a931aba9f3c5dff0856e6d3ed0712fbd38fe14094e20ec96ac87` |
| `tokenizer.json` | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |

The bundle pairs this export with ExecuTorch 1.5.0 and PyTorch 2.13.0+cpu
on portable ARM64 CPython 3.12.16. The archive includes the dependencies'
license notices. [ExecuTorch Pi documentation](https://docs.pytorch.org/executorch/stable/raspberry_pi_llama_tutorial.html) · [Portable Python build](https://github.com/astral-sh/python-build-standalone/releases/tag/20261026)
