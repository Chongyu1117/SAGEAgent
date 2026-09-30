"""A frozen chat LLM from the Hugging Face Hub. SAGEAgent never updates its weights."""

from __future__ import annotations

import torch


def _dtype_kwarg(dtype: torch.dtype) -> dict:
    """`from_pretrained` takes `dtype` in recent transformers and `torch_dtype` before 4.56."""
    import transformers
    from packaging.version import Version

    key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    return {key: dtype}


class ChatLLM:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, **_dtype_kwarg(getattr(torch, dtype)))
        self.model.eval().requires_grad_(False)

    def _render(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:  # chat templates without a system role
            merged = [{"role": "user", "content": f"{system}\n\n{user}"}]
            return self.tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def chat(self, system: str, users: list[str], temperature: float = 0.6, top_p: float = 0.9,
             max_new_tokens: int = 512, min_new_tokens: int = 0) -> list[str]:
        """Generate one reply per user message (batched)."""
        texts = [self._render(system, u) for u in users]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p} if temperature > 0 \
            else {"do_sample": False}
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                     pad_token_id=self.tokenizer.pad_token_id, **sampling)
        prompt_len = inputs["input_ids"].shape[1]
        return [self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in output]

    def generate(self, system: str, user: str, **kwargs) -> str:
        return self.chat(system, [user], **kwargs)[0]
