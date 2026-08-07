"""Environment-driven settings.

Every knob that a developer might want to sweep is an environment variable, so a
threshold study or an adapter comparison needs no code edit. That constraint is
what makes `make sweep` a one-liner instead of a fork.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class NIMSettings:
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "MVB_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
    )
    api_key: str | None = field(default_factory=lambda: os.environ.get("NVIDIA_API_KEY"))
    vlm_model: str = field(default_factory=lambda: os.environ.get("MVB_VLM_MODEL", "nvidia/vila"))
    llm_model: str = field(
        default_factory=lambda: os.environ.get(
            "MVB_LLM_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct"
        )
    )
    embed_model: str = field(
        default_factory=lambda: os.environ.get("MVB_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
    )
    timeout_s: float = field(default_factory=lambda: _env_float("MVB_NIM_TIMEOUT_S", 60.0))
    max_retries: int = field(default_factory=lambda: _env_int("MVB_NIM_MAX_RETRIES", 3))

    @property
    def enabled(self) -> bool:
        """True only when a key is present *and* the mock has not been forced.

        The forced-mock escape hatch exists because CI machines sometimes do have
        a key in the environment, and a test suite that silently starts making
        paid network calls is a bad surprise.
        """
        return bool(self.api_key) and not _env_bool("MVB_FORCE_MOCK", False)


@dataclass(frozen=True)
class GradingSettings:
    """Thresholds that turn a continuous score into a binary `correct`.

    Structured tasks are graded strictly because a downstream parser either works
    or does not. Free-form tasks get a similarity threshold, which is softer and
    correspondingly less trustworthy -- see the caveat in `docs/evaluation.md`.
    """

    freeform_threshold: float = field(
        default_factory=lambda: _env_float("MVB_FREEFORM_THRESHOLD", 0.62)
    )
    structured_threshold: float = field(
        default_factory=lambda: _env_float("MVB_STRUCTURED_THRESHOLD", 0.999)
    )
    judgement_requires_rationale: bool = field(
        default_factory=lambda: _env_bool("MVB_REQUIRE_RATIONALE", True)
    )


@dataclass(frozen=True)
class TrainSettings:
    """LoRA hyperparameters.

    The defaults are the configuration that won the sweep in
    `docs/post-training.md`, not library defaults. Rank 16 / alpha 32 on the
    attention projections plus the vision-language connector; freezing the
    connector cost ~4 points on spatial tasks, which is why it is trainable here.
    """

    adapter_name: str = field(default_factory=lambda: os.environ.get("MVB_ADAPTER", "lora-v3"))
    rank: int = field(default_factory=lambda: _env_int("MVB_LORA_RANK", 16))
    alpha: int = field(default_factory=lambda: _env_int("MVB_LORA_ALPHA", 32))
    dropout: float = field(default_factory=lambda: _env_float("MVB_LORA_DROPOUT", 0.05))
    learning_rate: float = field(default_factory=lambda: _env_float("MVB_LR", 1e-4))
    epochs: int = field(default_factory=lambda: _env_int("MVB_EPOCHS", 3))
    batch_size: int = field(default_factory=lambda: _env_int("MVB_BATCH_SIZE", 8))
    grad_accum: int = field(default_factory=lambda: _env_int("MVB_GRAD_ACCUM", 4))
    warmup_ratio: float = field(default_factory=lambda: _env_float("MVB_WARMUP_RATIO", 0.03))
    max_seq_len: int = field(default_factory=lambda: _env_int("MVB_MAX_SEQ_LEN", 2048))
    train_connector: bool = field(default_factory=lambda: _env_bool("MVB_TRAIN_CONNECTOR", True))
    seed: int = field(default_factory=lambda: _env_int("MVB_SEED", 1337))

    def target_modules(self) -> list[str]:
        mods = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if self.train_connector:
            mods.append("mm_projector")
        return mods


@dataclass(frozen=True)
class ObservabilitySettings:
    enabled: bool = field(default_factory=lambda: _env_bool("MVB_TRACE_ENABLED", True))
    trace_dir: str = field(default_factory=lambda: os.environ.get("MVB_TRACE_DIR", "runs/traces"))
    sample_rate: float = field(default_factory=lambda: _env_float("MVB_TRACE_SAMPLE_RATE", 1.0))
    redact_prompts: bool = field(default_factory=lambda: _env_bool("MVB_TRACE_REDACT", False))


@dataclass(frozen=True)
class Settings:
    nim: NIMSettings = field(default_factory=NIMSettings)
    grading: GradingSettings = field(default_factory=GradingSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    """Drop the cache. Tests mutate the environment between cases."""
    get_settings.cache_clear()
