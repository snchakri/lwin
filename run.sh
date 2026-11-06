#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 || $1 == --help || $1 == -h ]]; then
    echo 'Usage: bash run.sh {session|capsule} PASSAGE.txt [--require-offline]'
    exit 0
fi

case "$1" in
    session|capsule) mode=$1 ;;
    *) echo 'lwin: choose session or capsule.' >&2; exit 2 ;;
esac
shift
if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && $2 != --require-offline ) ]]; then
    echo 'Usage: bash run.sh {session|capsule} PASSAGE.txt [--require-offline]' >&2
    exit 2
fi
passage=$1
shift

if [[ $(uname -s) != Linux || $(uname -m) != aarch64 ]]; then
    echo 'lwin: this bundle requires 64-bit ARM Linux on a Raspberry Pi 5.' >&2
    exit 1
fi

bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python="$bundle/runtime/python/bin/python3"
native="$bundle/runtime/python/lib/python3.12/site-packages/executorch"
model="$bundle/model"
if [[ ! -x $python || ! -r $model/study-model.pte || ! -r $model/tokenizer.json ]]; then
    echo 'lwin: extract the full Raspberry Pi deployment archive before launching.' >&2
    exit 1
fi

# Use the bundled runtime and matching operators so the Pi can start offline.
export PYTHONPATH="$bundle/src"
export LD_LIBRARY_PATH="$bundle/runtime/lib"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset PYTHONHOME PYTHONPYCACHEPREFIX PYTHONPLATLIBDIR
exec "$python" -s -B -m lwin.cli --"$mode" --source-recall \
    --model "$model/study-model.pte" --tokenizer "$model/tokenizer.json" \
    --model-id Qwen/Qwen2.5-0.5B-Instruct --checkpoint 7ae557604adf67be50417f59c2c2f167def9a775 \
    --backend XNNPACK --quantisation '8da4w GPTQ' \
    --prompt-template "$model/chat-template.txt" --end-marker '<|im_end|>' \
    --kernel-library "$native/lib/libexecutorch_kernels_quantized.so" \
    --kernel-library "$native/extension/llm/custom_ops/libcustom_ops_aot_lib.so" "$@" -- "$passage"
