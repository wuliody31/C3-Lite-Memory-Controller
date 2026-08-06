#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import (
    BaseHTTPRequestHandler,
    ThreadingHTTPServer,
)
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    import torch.backends.python_native as python_native

    python_native.triton.enabled = False
except (ImportError, AttributeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=18080,
    )
    parser.add_argument(
        "--request-log",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def flatten_content(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts: list[str] = []

        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")

                if isinstance(text, str):
                    parts.append(text)

        return "\n".join(parts)

    return str(value)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()

    if stripped.startswith("```"):
        lines = stripped.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        stripped = "\n".join(lines).strip()

    try:
        value = json.loads(stripped)

        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")

    if start < 0:
        raise ValueError(
            "No JSON object found in model response."
        )

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(stripped)):
        char = stripped[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False

            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                candidate = stripped[start:index + 1]
                value = json.loads(candidate)

                if not isinstance(value, dict):
                    raise ValueError(
                        "Extracted JSON is not an object."
                    )

                return value

    raise ValueError(
        "Could not find a balanced JSON object."
    )


class QwenRuntime:
    def __init__(
        self,
        *,
        model_id: str,
    ) -> None:
        self.model_id = model_id
        self.lock = threading.Lock()

        print(
            f"Loading tokenizer: {model_id}",
            flush=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
            trust_remote_code=False,
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print(
            f"Loading model: {model_id}",
            flush=True,
        )

        started = time.perf_counter()

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                local_files_only=True,
                trust_remote_code=False,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="sdpa",
            )
        )

        self.model.eval()

        elapsed = time.perf_counter() - started

        print(
            f"Model loaded in {elapsed:.3f}s",
            flush=True,
        )

        print(
            "GPU memory allocated GiB:",
            round(
                torch.cuda.memory_allocated()
                / 1024**3,
                3,
            ),
            flush=True,
        )

    def _prepare_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool,
        retry: bool,
    ) -> list[dict[str, str]]:
        prepared: list[dict[str, str]] = []

        for message in messages:
            prepared.append(
                {
                    "role": str(
                        message.get("role", "user")
                    ),
                    "content": flatten_content(
                        message.get("content", "")
                    ),
                }
            )

        if json_mode:
            instruction = (
                "\n\nReturn only one valid JSON object. "
                "Do not use Markdown fences, commentary, "
                "or text outside the JSON object."
            )

            if retry:
                instruction += (
                    " Your previous response was not "
                    "valid JSON, so strictly follow the "
                    "requested JSON schema."
                )

            for message in reversed(prepared):
                if message["role"] == "user":
                    message["content"] += instruction
                    break
            else:
                prepared.append(
                    {
                        "role": "user",
                        "content": instruction.strip(),
                    }
                )

        return prepared

    def _generate_once(
        self,
        *,
        messages: list[dict[str, Any]],
        json_mode: bool,
        retry: bool,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: Any,
    ) -> tuple[str, int, int]:
        prepared = self._prepare_messages(
            messages,
            json_mode=json_mode,
            retry=retry,
        )

        prompt = self.tokenizer.apply_chat_template(
            prepared,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        input_tokens = int(
            inputs["input_ids"].shape[-1]
        )

        do_sample = temperature > 1e-8

        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max(
                1,
                min(int(max_tokens), 1024),
            ),
            "do_sample": do_sample,
            "use_cache": True,
            "eos_token_id": (
                self.tokenizer.eos_token_id
            ),
            "pad_token_id": (
                self.tokenizer.pad_token_id
            ),
        }

        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": max(
                        float(temperature),
                        1e-5,
                    ),
                    "top_p": float(top_p),
                }
            )

        with self.lock:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **generation_kwargs
                )

        new_ids = output_ids[
            0,
            input_tokens:,
        ]

        output_tokens = int(new_ids.numel())

        response = self.tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
        ).strip()

        stop_sequences: list[str] = []

        if isinstance(stop, str):
            stop_sequences = [stop]
        elif isinstance(stop, list):
            stop_sequences = [
                str(item)
                for item in stop
                if item
            ]

        for sequence in stop_sequences:
            position = response.find(sequence)

            if position >= 0:
                response = response[:position]
                break

        return (
            response.strip(),
            input_tokens,
            output_tokens,
        )

    def generate(
        self,
        payload: dict[str, Any],
    ) -> tuple[str, int, int, str]:
        if payload.get("tools"):
            raise ValueError(
                "Tools are not supported by this "
                "Phase II-A protocol smoke server."
            )

        messages = payload.get("messages")

        if not isinstance(messages, list):
            raise ValueError(
                "'messages' must be a list."
            )

        response_format = payload.get(
            "response_format"
        )

        json_mode = (
            isinstance(response_format, dict)
            and response_format.get("type")
            == "json_object"
        )

        temperature = float(
            payload.get("temperature", 0.0)
            or 0.0
        )

        top_p = float(
            payload.get("top_p", 1.0)
            or 1.0
        )

        max_tokens = int(
            payload.get(
                "max_tokens",
                payload.get(
                    "max_completion_tokens",
                    512,
                ),
            )
            or 512
        )

        last_response = ""

        attempts = 2 if json_mode else 1

        for attempt in range(attempts):
            response, input_tokens, output_tokens = (
                self._generate_once(
                    messages=messages,
                    json_mode=json_mode,
                    retry=attempt > 0,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=payload.get("stop"),
                )
            )

            last_response = response

            if not json_mode:
                return (
                    response,
                    input_tokens,
                    output_tokens,
                    response,
                )

            try:
                value = extract_json_object(
                    response
                )

                canonical = json.dumps(
                    value,
                    ensure_ascii=False,
                )

                return (
                    canonical,
                    input_tokens,
                    output_tokens,
                    response,
                )
            except (
                json.JSONDecodeError,
                ValueError,
            ):
                if attempt + 1 == attempts:
                    raise ValueError(
                        "Model failed to return valid "
                        "JSON after two attempts. "
                        f"Last response: {last_response!r}"
                    )

        raise RuntimeError(
            "Generation loop terminated unexpectedly."
        )


class RequestHandler(BaseHTTPRequestHandler):
    runtime: QwenRuntime
    request_log: Path
    log_lock = threading.Lock()

    def log_message(
        self,
        format: str,
        *args: Any,
    ) -> None:
        print(
            "%s - %s"
            % (
                self.address_string(),
                format % args,
            ),
            flush=True,
        )

    def _write_json(
        self,
        status: int,
        value: dict[str, Any],
    ) -> None:
        body = json.dumps(
            value,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json",
        )
        self.send_header(
            "Content-Length",
            str(len(body)),
        )
        self.end_headers()
        self.wfile.write(body)

    def _log_event(
        self,
        value: dict[str, Any],
    ) -> None:
        self.request_log.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.log_lock:
            with self.request_log.open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(
                200,
                {
                    "status": "ok",
                    "model": (
                        self.runtime.model_id
                    ),
                },
            )
            return

        if self.path == "/v1/models":
            self._write_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": (
                                self.runtime.model_id
                            ),
                            "object": "model",
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return

        self._write_json(
            404,
            {
                "error": {
                    "message": "Not found"
                }
            },
        )

    def do_POST(self) -> None:
        request_id = (
            "chatcmpl-"
            + uuid.uuid4().hex
        )

        if self.path != "/v1/chat/completions":
            self._write_json(
                404,
                {
                    "error": {
                        "message": "Not found"
                    }
                },
            )
            return

        try:
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            raw_body = self.rfile.read(
                content_length
            )

            payload = json.loads(
                raw_body.decode("utf-8")
            )

            started = time.perf_counter()

            (
                content,
                prompt_tokens,
                completion_tokens,
                raw_model_response,
            ) = self.runtime.generate(payload)

            elapsed = (
                time.perf_counter()
                - started
            )

            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get(
                    "model",
                    self.runtime.model_id,
                ),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": (
                        prompt_tokens
                    ),
                    "completion_tokens": (
                        completion_tokens
                    ),
                    "total_tokens": (
                        prompt_tokens
                        + completion_tokens
                    ),
                },
            }

            self._log_event(
                {
                    "request_id": request_id,
                    "timestamp": time.time(),
                    "request": payload,
                    "raw_model_response": (
                        raw_model_response
                    ),
                    "returned_content": (
                        content
                    ),
                    "elapsed_seconds": elapsed,
                    "status": "success",
                }
            )

            self._write_json(
                200,
                response,
            )

        except Exception as exc:
            self._log_event(
                {
                    "request_id": request_id,
                    "timestamp": time.time(),
                    "status": "error",
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error": str(exc),
                }
            )

            self._write_json(
                500,
                {
                    "error": {
                        "message": str(exc),
                        "type": (
                            type(exc).__name__
                        ),
                    }
                },
            )


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable."
        )

    runtime = QwenRuntime(
        model_id=args.model
    )

    RequestHandler.runtime = runtime
    RequestHandler.request_log = (
        args.request_log
    )

    server = ThreadingHTTPServer(
        (args.host, args.port),
        RequestHandler,
    )

    print(
        f"Server ready: "
        f"http://{args.host}:{args.port}",
        flush=True,
    )

    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
