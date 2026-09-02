# PAN 2026 — Qwen3-0.6B + QLoRA

Advanced model submission for the PAN 2026 Voight-Kampff AI Detection task.
Qwen3-0.6B base model with QLoRA adapters; base + adapters hosted on HuggingFace
at `eljosey40/qwen3-lora-pan26-voightkampff`.

Submission by Jose Alejandro Perez Dominguez
Master en Inteligencia Artificial — Universidad Europea de Valencia

## Usage

    python predict.py /path/to/dataset.jsonl /path/to/output_dir

Output: `predictions.jsonl` with `{"id": "...", "score": 0.XXXX}` per line.
Score > 0.5 = AI-generated, score < 0.5 = human-written.
