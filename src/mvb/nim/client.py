"""NIM client.

NIM exposes an OpenAI-compatible surface, which is the single most useful fact
about it for a blueprint like this: the same client code points at
`https://integrate.api.nvidia.com/v1` for the hosted catalog or at
`http://localhost:8000/v1` for a self-hosted NIM container, and nothing else
changes. Every deployment decision in `docs/architecture.md` follows from that.
"""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..schemas import Example, Frame, Prediction
from .mock import MockVLMBackend


def encode_frame(frame: Frame) -> str:
    """Return a data URI for a local frame, or pass through a remote URL.

    NIM's vision endpoints accept both. Local files are inlined because a
    blueprint that requires the user to stand up object storage before the first
    request is a blueprint nobody finishes.
    """
    if frame.uri.startswith(("http://", "https://", "data:")):
        return frame.uri
    path = Path(frame.uri)
    if not path.exists():
        raise FileNotFoundError(f"frame not found: {frame.uri}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{payload}"


def build_messages(ex: Example) -> list[dict[str, Any]]:
    """Assemble the chat payload for one example.

    The system prompt is task-conditional and terse. Long system prompts were the
    first thing the failure analysis showed to be counterproductive: the base
    model followed the last instruction it saw and ignored the rest.
    """
    from ..schemas import TaskKind

    instructions = {
        TaskKind.SCENE_QA: "Answer the question about the image in one or two sentences. State only what is visible.",
        TaskKind.ATTRIBUTE_EXTRACTION: "Return a single JSON object and nothing else. No prose, no code fences.",
        TaskKind.SPATIAL_RELATION: "Answer with the spatial relation only. Be precise about left/right and near/far.",
        TaskKind.ANOMALY_JUDGEMENT: "Give a verdict, then justify it with concrete visual evidence in the same sentence.",
        TaskKind.TEMPORAL_ORDERING: "Return a JSON array of event labels in chronological order. Nothing else.",
    }
    content: list[dict[str, Any]] = [{"type": "text", "text": ex.prompt}]
    for frame in ex.frames:
        content.append({"type": "image_url", "image_url": {"url": encode_frame(frame)}})
    return [
        {"role": "system", "content": instructions[ex.task]},
        {"role": "user", "content": content},
    ]


class NIMBackend:
    """Live backend. Only constructed when a key is present."""

    name = "nim"

    def __init__(self) -> None:
        import httpx  # imported lazily so the mock path has no hard dependency

        cfg = get_settings().nim
        self.cfg = cfg
        self._client = httpx.Client(
            base_url=cfg.base_url,
            timeout=cfg.timeout_s,
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Accept": "application/json",
            },
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        last: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._client.post(path, json=payload)
                if resp.status_code == 429 or resp.status_code >= 500:
                    # Retry only on throttling and server faults. A 400 means the
                    # payload is wrong and retrying it just wastes the quota.
                    raise httpx.HTTPStatusError(
                        f"retryable {resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last = exc
                if attempt == self.cfg.max_retries - 1:
                    break
                time.sleep(min(8.0, 0.75 * (2**attempt)))
        raise RuntimeError(f"NIM request to {path} failed after retries: {last}") from last

    def predict(self, ex: Example, adapter: str = "base") -> Prediction:
        model = self.cfg.vlm_model
        if adapter and adapter != "base":
            # NIM serves LoRA adapters as model-name suffixes when the container
            # is started with --lora-modules. Hosted catalog endpoints ignore it.
            model = f"{model}:{adapter}"

        t0 = time.perf_counter()
        data = self._post(
            "/chat/completions",
            {
                "model": model,
                "messages": build_messages(ex),
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 512,
            },
        )
        latency = (time.perf_counter() - t0) * 1000.0
        usage = data.get("usage", {})
        return Prediction(
            example_id=ex.id,
            task=ex.task,
            raw=data["choices"][0]["message"]["content"],
            latency_ms=latency,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            adapter=adapter,
        )

    def chat(self, prompt: str, **kwargs: Any) -> str:
        data = self._post(
            "/chat/completions",
            {
                "model": self.cfg.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.0),
                "max_tokens": kwargs.get("max_tokens", 512),
            },
        )
        return data["choices"][0]["message"]["content"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post(
            "/embeddings",
            {"model": self.cfg.embed_model, "input": texts, "input_type": "passage"},
        )
        return [row["embedding"] for row in data["data"]]


_BACKEND: Any = None


def get_backend(force_mock: bool = False) -> Any:
    """Return the live backend if configured, otherwise the mock.

    Falling back silently is the right call for a blueprint: the failure mode we
    want to avoid is a developer cloning the repo and hitting an auth error
    before they have seen anything work.
    """
    global _BACKEND
    if _BACKEND is not None and not force_mock:
        return _BACKEND
    if force_mock or not get_settings().nim.enabled:
        _BACKEND = MockVLMBackend()
    else:
        _BACKEND = NIMBackend()
    return _BACKEND


def reset_backend() -> None:
    global _BACKEND
    _BACKEND = None
