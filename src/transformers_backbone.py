from __future__ import annotations

from pathlib import Path
from typing import Any

from .backbones import GenerationResult

from .errors import GenerationError


class TransformersBackbone:
    """Frozen Hugging Face Transformers backbone for C3-Lite.

    The model is loaded once when the backbone is created and then reused for
    all queries. This is important for benchmark efficiency and for keeping
    the same frozen model across C3-Lite and baseline methods.
    """

    SUPPORTED_DTYPES = {
        "float16",
        "bfloat16",
        "float32",
    }

    def __init__(
        self,
        *,
        model: str,
        load_in_4bit: bool = True,
        compute_dtype: str = "bfloat16",
        enable_thinking: bool = False,
        adapter_path: str | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("A Hugging Face model name is required.")

        if compute_dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(
                "compute_dtype must be one of: "
                + ", ".join(sorted(self.SUPPORTED_DTYPES))
            )

        if adapter_path and not Path(adapter_path).exists():
            raise FileNotFoundError(
                f"LoRA adapter directory does not exist: {adapter_path}"
            )

        # Imports are intentionally lazy. Existing mock/controller tests can
        # run without loading PyTorch, CUDA, Transformers, or model weights.
        import torch

        # Cluster compatibility workaround for PyTorch 2.13.
        # It prevents Triton from trying to compile a runtime extension that
        # requires unavailable Python development headers.
        try:
            from torch._native.registry import deregister_op_overrides

            deregister_op_overrides(
                disable_op_symbols="bmm"
            )
            print(
                "Disabled PyTorch native BMM override; "
                "using ATen fallback."
            )
        except Exception as exc:
            print(
                "BMM compatibility patch was not applied: "
                f"{exc}"
            )

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        dtype = getattr(torch, compute_dtype)

        quantization_config = None

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )

        tokenizer_source = adapter_path or model

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                use_fast=True,
            )
        except Exception:
            # Some adapters do not include tokenizer files.
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                use_fast=True,
            )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, object] = {
            "device_map": "auto",
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = (
                quantization_config
            )

        loaded_model: Any = (
            AutoModelForCausalLM.from_pretrained(
                model,
                **model_kwargs,
            )
        )

        if adapter_path:
            from peft import PeftModel

            loaded_model = PeftModel.from_pretrained(
                loaded_model,
                adapter_path,
                is_trainable=False,
            )

        loaded_model.eval()

        self.model_name = model
        self.adapter_path = adapter_path
        self.enable_thinking = bool(enable_thinking)
        self.tokenizer: Any = tokenizer
        self.model: Any = loaded_model
        self.torch = torch

        try:
            self.device = next(
                self.model.parameters()
            ).device
        except StopIteration as exc:
            raise GenerationError(
                "The loaded model contains no parameters."
            ) from exc

        print(f"Loaded backbone: {self.model_name}")
        print(f"Backbone device: {self.device}")
        print(
            "LoRA adapter:",
            self.adapter_path or "none",
        )

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> GenerationResult:
        if not prompt.strip():
            raise ValueError(
                "The generation prompt must not be empty."
            )

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        rendered_prompt = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        )

        model_inputs = self.tokenizer(
            rendered_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        model_inputs = {
            key: value.to(self.device)
            for key, value in model_inputs.items()
        }

        input_tokens = int(
            model_inputs["input_ids"].shape[-1]
        )

        do_sample = float(temperature) > 0.0

        generation_kwargs: dict[str, object] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Do not pass temperature during greedy decoding.
        if do_sample:
            generation_kwargs["temperature"] = float(
                temperature
            )

        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **model_inputs,
                **generation_kwargs,
            )

        new_token_ids = output_ids[
            0,
            input_tokens:,
        ]

        generated_text = self.tokenizer.decode(
            new_token_ids,
            skip_special_tokens=True,
        ).strip()

        if not generated_text:
            raise GenerationError(
                "The Transformers backbone returned "
                "an empty generated response."
            )

        return GenerationResult(
            text=generated_text,
            input_tokens=input_tokens,
            output_tokens=int(new_token_ids.numel()),
        )
