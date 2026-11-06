# lwin

lwin helps learners review a short passage without internet access. It builds
a summary and three questions, keeps each answer linked to the original text,
and offers prediction, reflection, and one round of feedback in the terminal.

## Run on Raspberry Pi

Use a Raspberry Pi 5 with at least 4 GiB RAM, 64-bit Raspberry Pi OS Trixie, and
at least 2 GiB of free space. Copy the archive to the Pi and extract it in your
home directory as a regular user. It includes ARM64 Python, ExecuTorch, the
model, and its tokenizer. [Download the Raspberry Pi deployment bundle](https://github.com/snchakri/lwin/releases/download/v0.1.0/lwin-raspberry-pi-arm64.tar.gz)

```bash
tar -xzf lwin-raspberry-pi-arm64.tar.gz
cd lwin
bash run.sh session /path/to/passage.txt
```

Use a UTF-8 passage of 150–300 words. For an answer key with source offsets and
run details, use `bash run.sh capsule /path/to/passage.txt > study-aid.json`.
For a disconnected run, use the Pi's local terminal, turn off Wi-Fi and unplug
Ethernet, then add `--require-offline` to either command. If an interface is
still up, check its name with `ip -br link` and bring it down with
`sudo ip link set <interface> down`. Inference always reads local assets.

## Follow the code

1. [The command line](src/lwin/cli.py) reads the passage and picks the learner
   session or teacher export.
2. [Study aid generation](src/lwin/study_aid.py) asks the local model to pick
   source sentences and quiz terms. [Validation](src/lwin/validation.py) and
   [capsule assembly](src/lwin/capsule.py) verify and record the original spans.
3. [The learner session](src/lwin/learner_session.py) controls what appears
   before and after a prediction. [The tutor](src/lwin/tutor.py) uses the same
   model to assess a response and choose one next step. Learner text stays in
   memory for that session.
4. [The model worker](src/lwin/runtime_worker.py) keeps one native runner alive
   for the session and can stop it if a request stalls.

The included model produces extractive summaries and fill-in-the-blank
questions. A summary can miss a central detail. The tutor can ask for
clarification when its model is unsure about a definite wrong answer. Review
the teacher export before using it with a class. [Model details](model/README.md)
