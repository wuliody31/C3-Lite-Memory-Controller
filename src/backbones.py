from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class GenerationResult:
    """Text and optional token usage returned by a frozen backbone."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class Backbone(Protocol):
    """Common interface implemented by all C3-Lite backbones."""

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> GenerationResult:
        ...


class MockBackbone:
    """Deterministic adapter for smoke tests only.

    It verifies that selected memory identifiers were inserted into the
    LLM-facing prompt. It is not a language model and must not be reported as
    an experimental backbone.

    RC3 prompts deliberately omit internal score/status metadata, so evidence
    headers now look like ``[p_user01_001]`` or
    ``[s_user01_003] (time=...; role=current_endpoint)``. The regular
    expression below supports both forms.
    """

    MEMORY_HEADER_PATTERN = re.compile(
        r"^\[((?:e|s|p)_[A-Za-z0-9_.:-]+)\](?:\s|\(|$)"
    )

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> GenerationResult:
        del temperature
        del max_new_tokens

        memory_ids: list[str] = []
        for raw_line in prompt.splitlines():
            line = raw_line.strip()
            match = self.MEMORY_HEADER_PATTERN.match(line)
            if match:
                memory_ids.append(match.group(1))

        # Preserve prompt order while removing duplicates.
        memory_ids = list(dict.fromkeys(memory_ids))

        if not memory_ids:
            return GenerationResult(
                text=(
                    "Pipeline smoke test completed, but no selected memory "
                    "IDs were found in the generated prompt."
                )
            )

        return GenerationResult(
            text=(
                "Pipeline smoke test completed. Selected memories: "
                + ", ".join(memory_ids)
                + ". Use Ollama or the GPU backbone for the final answer."
            )
        )


class OllamaBackbone:
    """Frozen local language-model adapter using Ollama's generate API."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 180,
    ) -> None:
        if not model.strip():
            raise ValueError("An Ollama model name is required.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> GenerationResult:
        if not prompt.strip():
            raise ValueError("The generation prompt must not be empty.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_new_tokens),
            },
        }

        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                response_data = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code} at {self.base_url}: "
                f"{details or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Unable to call Ollama at {self.base_url}. Check that "
                "Ollama is running and the requested model is installed."
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama generation timed out after {self.timeout} seconds."
            ) from exc

        try:
            data = json.loads(response_data)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Ollama returned a response that was not valid JSON."
            ) from exc

        generated_text = str(data.get("response", "")).strip()
        if not generated_text:
            error_message = str(data.get("error", "")).strip()
            if error_message:
                raise RuntimeError(f"Ollama generation failed: {error_message}")
            raise RuntimeError("Ollama returned an empty generated response.")

        return GenerationResult(
            text=generated_text,
            input_tokens=_optional_int(data.get("prompt_eval_count")),
            output_tokens=_optional_int(data.get("eval_count")),
        )


def _optional_int(value: object) -> int | None:
    """Convert optional Ollama token counts without raising on bad metadata."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None