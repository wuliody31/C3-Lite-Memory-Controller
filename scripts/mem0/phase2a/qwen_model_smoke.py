#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import time

import torch

# PHASE2A_DISABLE_PYTHON_NATIVE_TRITON
# The cluster compute image does not provide Python.h, which Triton's
# runtime helper compilation requires. Disable PyTorch's optional
# Python-native Triton implementations and use regular CUDA kernels.
import torch.backends.python_native as python_native

python_native.triton.enabled = False
from transformers.utils.hub import cached_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_ID = "Qwen/Qwen3-8B"


def gib(value: int) -> float:
    return value / 1024**3


def gpu_snapshot(label: str) -> None:
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    print()
    print(f"===== GPU MEMORY: {label} =====")
    print(f"allocated GiB:      {gib(allocated):.3f}")
    print(f"reserved GiB:       {gib(reserved):.3f}")
    print(f"peak allocated GiB: {gib(peak_allocated):.3f}")
    print(f"peak reserved GiB:  {gib(peak_reserved):.3f}")


def main() -> None:
    print("========================================")
    print("PHASE II-A QWEN3 FULL MODEL SMOKE")
    print("========================================")
    print("Host:", os.uname().nodename)
    print("Model:", MODEL_ID)
    print("Torch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable inside the GPU job."
        )

    print("GPU:", torch.cuda.get_device_name(0))

    config_path = cached_file(
        MODEL_ID,
        "config.json",
        local_files_only=True,
    )

    print("Resolved config path:", config_path)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        trust_remote_code=False,
    )

    tokenizer_seconds = time.perf_counter() - tokenizer_start

    print(
        f"Tokenizer load seconds: "
        f"{tokenizer_seconds:.3f}"
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(
        "Quantization config:",
        json.dumps(
            quantization_config.to_dict(),
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
    )

    model_start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa",
    )

    model.eval()

    model_seconds = time.perf_counter() - model_start

    print(
        f"Model load seconds: "
        f"{model_seconds:.3f}"
    )
    print("Model class:", model.__class__.__name__)
    print("Model device:", model.device)

    gpu_snapshot("after model load")

    messages = [
        {
            "role": "user",
            "content": (
                "Return exactly the word READY "
                "and nothing else."
            ),
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print()
    print("Rendered prompt:")
    print(repr(prompt))

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        name: tensor.to(model.device)
        for name, tensor in inputs.items()
    }

    input_token_count = int(
        inputs["input_ids"].shape[-1]
    )

    generation_start = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generation_seconds = (
        time.perf_counter() - generation_start
    )

    new_token_ids = output_ids[
        0,
        input_token_count:,
    ]

    response = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True,
    ).strip()

    print()
    print("===== GENERATION RESULT =====")
    print("Input tokens:", input_token_count)
    print("Output tokens:", int(new_token_ids.numel()))
    print(
        f"Generation seconds: "
        f"{generation_seconds:.3f}"
    )
    print("Raw response:", repr(response))

    gpu_snapshot("after generation")

    normalized = response.upper().strip()

    if normalized != "READY":
        raise RuntimeError(
            "Unexpected response. "
            f"Expected 'READY', received {response!r}."
        )

    print()
    print(
        "PHASE II-A QWEN3 FULL MODEL SMOKE: PASSED"
    )


if __name__ == "__main__":
    main()
