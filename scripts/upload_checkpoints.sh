#!/bin/bash
# Publish the two student checkpoints to the Hugging Face Hub, from a machine that holds them
# (the cluster's login node has them under $SCRATCH/students/ and has internet).
#
#   pip install -U huggingface_hub && hf auth login          # once, with a write token
#   HF_ORG=<your org or user> bash scripts/upload_checkpoints.sh
#
# Each repo gets the checkpoint's `best/` directory (weights, config, tokenizer), a
# claimprune.json with the model's keep threshold, and MODEL_CARD.md as README.md.
set -euo pipefail
ORG="${HF_ORG:?set HF_ORG to your Hugging Face org or user name}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
STUDENTS="${STUDENTS_DIR:-$SCRATCH/students}"
declare -A SRC=( ["claimprune-qwen3.5-2b"]="$STUDENTS/2b_full_f1.0_21774288/best"
                 ["claimprune-modernbert-large"]="$STUDENTS/enc_ModernBERT-large_21845184/best" )
declare -A THR=( ["claimprune-qwen3.5-2b"]="0.3073580265045166"
                 ["claimprune-modernbert-large"]="0.3458289667963982" )
declare -A BASE=( ["claimprune-qwen3.5-2b"]="Qwen/Qwen3.5-2B"
                  ["claimprune-modernbert-large"]="answerdotai/ModernBERT-large" )
for name in "${!SRC[@]}"; do
    d="${SRC[$name]}"
    [ -f "$d/config.json" ] || { echo "FATAL: no checkpoint at $d"; exit 1; }
    stage="$(mktemp -d)"
    cp -r "$d"/. "$stage"/
    printf '{"threshold": %s, "package": "claimprune", "window_sentences": 48, "window_chars": 10000, "cap_windows": 16, "stop_after_empty": 2}\n' "${THR[$name]}" > "$stage/claimprune.json"
    sed -e "s#<Qwen/Qwen3.5-2B | answerdotai/ModernBERT-large>#${BASE[$name]}#" \
        -e "s#claimprune-<qwen3.5-2b | modernbert-large>#$name#" -e "s#<this repo id>#$ORG/$name#" \
        "$HERE/MODEL_CARD.md" > "$stage/README.md"
    echo "== $ORG/$name  <-  $d  ($(du -sh "$stage" | cut -f1))"
    hf repo create "$ORG/$name" --type model -y >/dev/null 2>&1 || true
    hf upload "$ORG/$name" "$stage" . --commit-message "claimprune $name: checkpoint, threshold, model card"
    rm -rf "$stage"
done
echo "done. Test on the laptop:  python -c \"from claimprune import Pruner; print(Pruner('$ORG/claimprune-modernbert-large').threshold)\""
