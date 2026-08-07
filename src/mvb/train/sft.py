"""SFT / LoRA post-training.

Two things live here, and it is worth being clear about which is which.

**Real:** the data formatting, the prompt/completion masking, the LoRA target
selection, the config surface, and the run manifest. These run with no GPU and
are what you would actually keep when swapping in a different trainer.

**Requires a GPU:** `train()` itself, which imports `peft` and `transformers`
lazily and raises a useful message when they are absent. The repository does not
ship checkpoints -- the mock backend in `nim/mock.py` stands in for the trained
adapters so the evaluation and serving paths stay exercisable end to end.

The masking detail below is the one that is easy to get wrong and expensive to
notice: if the prompt tokens are not masked out of the loss, the model spends
most of its gradient learning to reproduce your instruction template, and the
run looks fine because the loss still goes down.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import TrainSettings, get_settings
from ..schemas import Example, Split, TaskKind

IGNORE_INDEX = -100

_TASK_INSTRUCTION = {
    TaskKind.SCENE_QA: "Answer the question about the image in one or two sentences. State only what is visible.",
    TaskKind.ATTRIBUTE_EXTRACTION: "Return a single JSON object and nothing else. No prose, no code fences.",
    TaskKind.SPATIAL_RELATION: "Answer with the spatial relation only. Be precise about left/right and near/far.",
    TaskKind.ANOMALY_JUDGEMENT: "Give a verdict, then justify it with concrete visual evidence in the same sentence.",
    TaskKind.TEMPORAL_ORDERING: "Return a JSON array of event labels in chronological order. Nothing else.",
}


@dataclass
class FormattedExample:
    prompt: str
    completion: str
    n_images: int
    task: str
    example_id: str

    def text(self) -> str:
        return self.prompt + self.completion


def format_example(ex: Example, include_rationale: bool = True) -> FormattedExample:
    """Render one example into a prompt/completion pair.

    `include_rationale` is a real lever, not a flag for symmetry: training the
    judgement task on verdict-only completions produced a model that answered
    "Yes" with no justification, which the grader scores at 0.7 -- below
    threshold. Rationale supervision is the v1 -> v2 change.
    """
    instruction = _TASK_INSTRUCTION[ex.task]
    image_tokens = "".join("<image>\n" for _ in ex.frames)
    prompt = f"<|system|>\n{instruction}\n<|user|>\n{image_tokens}{ex.prompt}\n<|assistant|>\n"

    completion = ex.target
    if include_rationale and ex.rationale and ex.task is TaskKind.ANOMALY_JUDGEMENT:
        completion = f"{ex.target}. Because {ex.rationale}"
    completion += "<|end|>"

    return FormattedExample(
        prompt=prompt,
        completion=completion,
        n_images=len(ex.frames),
        task=ex.task.value,
        example_id=ex.id,
    )


def build_labels(prompt_ids: list[int], completion_ids: list[int]) -> list[int]:
    """Mask the prompt out of the loss.

    Without this the model is trained to generate the instruction template as
    well as the answer. The failure is quiet: loss decreases normally, the model
    just spends capacity on text it will always be given for free.
    """
    return [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)


def to_jsonl(examples: list[Example], path: str | Path, include_rationale: bool = True) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fmt = format_example(ex, include_rationale=include_rationale)
            fh.write(
                json.dumps(
                    {
                        "id": fmt.example_id,
                        "task": fmt.task,
                        "images": [f.uri for f in ex.frames],
                        "prompt": fmt.prompt,
                        "completion": fmt.completion,
                    }
                )
                + "\n"
            )
            n += 1
    return n


@dataclass
class RunManifest:
    """Everything needed to explain a number in the results table six months on.

    Recording the *data* fingerprint alongside the hyperparameters is the part
    people skip, and it is the part that matters -- most unreproducible results
    in this repo's history were a changed dataset, not a changed learning rate.
    """

    adapter: str
    settings: dict[str, Any]
    n_train: int
    n_val: int
    task_counts: dict[str, int]
    provenance_counts: dict[str, int]
    started_at: float = field(default_factory=time.time)
    host: str = field(default_factory=platform.platform)
    notes: str = ""

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


def build_manifest(
    examples: list[Example], cfg: TrainSettings | None = None, notes: str = ""
) -> RunManifest:
    from collections import Counter

    cfg = cfg or get_settings().train
    train = [e for e in examples if e.split is Split.TRAIN]
    val = [e for e in examples if e.split is Split.VAL]
    return RunManifest(
        adapter=cfg.adapter_name,
        settings=asdict(cfg) | {"target_modules": cfg.target_modules()},
        n_train=len(train),
        n_val=len(val),
        task_counts=dict(Counter(e.task.value for e in train)),
        provenance_counts=dict(Counter(e.provenance.value for e in train)),
        notes=notes,
    )


def lora_config(cfg: TrainSettings | None = None) -> dict[str, Any]:
    """The peft LoraConfig kwargs, as plain data so they can be asserted on."""
    cfg = cfg or get_settings().train
    return {
        "r": cfg.rank,
        "lora_alpha": cfg.alpha,
        "lora_dropout": cfg.dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": cfg.target_modules(),
    }


def train(
    examples: list[Example],
    output_dir: str | Path = "checkpoints",
    cfg: TrainSettings | None = None,
) -> Path:  # pragma: no cover - requires a GPU and the training extra
    """Run the LoRA fine-tune. Requires `pip install 'mvb[train]'` and a GPU."""
    cfg = cfg or get_settings().train
    try:
        import torch  # noqa: F401
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForVision2Seq,
            AutoProcessor,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "training requires the optional extra: pip install 'mvb[train]'\n"
            "The evaluation and serving paths do not need it -- they run on the mock backend."
        ) from exc

    out = Path(output_dir) / cfg.adapter_name
    out.mkdir(parents=True, exist_ok=True)

    base = get_settings().nim.vlm_model
    processor = AutoProcessor.from_pretrained(base)
    model = AutoModelForVision2Seq.from_pretrained(base, torch_dtype="auto", device_map="auto")
    model = get_peft_model(model, LoraConfig(**lora_config(cfg)))
    model.print_trainable_parameters()

    train_rows = [format_example(e) for e in examples if e.split is Split.TRAIN]
    val_rows = [format_example(e) for e in examples if e.split is Split.VAL]

    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        seed=cfg.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_TorchDataset(train_rows, processor, cfg.max_seq_len),
        eval_dataset=_TorchDataset(val_rows, processor, cfg.max_seq_len),
    )
    trainer.train()
    model.save_pretrained(out)
    build_manifest(examples, cfg).save(out / "manifest.json")
    return out


class _TorchDataset:  # pragma: no cover - only constructed on the training path
    """Applies `build_labels` so the prompt is excluded from the loss."""

    def __init__(self, rows: list[FormattedExample], processor: Any, max_len: int) -> None:
        self.rows = rows
        self.processor = processor
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.rows[i]
        tok = self.processor.tokenizer
        prompt_ids = tok(row.prompt, add_special_tokens=False).input_ids
        completion_ids = tok(row.completion, add_special_tokens=False).input_ids
        input_ids = (prompt_ids + completion_ids)[: self.max_len]
        labels = build_labels(prompt_ids, completion_ids)[: self.max_len]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}
