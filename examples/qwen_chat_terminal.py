"""Small interactive terminal for chatting with a local Hugging Face Qwen model."""

from __future__ import annotations

import argparse
import sys
from threading import Thread


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    print(f"Loading {args.model} on {args.device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    messages: list[dict[str, str]] = [{"role": "system", "content": args.system}]
    print("Ready. Commands: /reset, /help, /exit", flush=True)

    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.", flush=True)
            return 0
        if not prompt:
            continue
        if prompt == "/exit":
            print("Exiting.", flush=True)
            return 0
        if prompt == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print("Conversation reset.", flush=True)
            continue
        if prompt == "/help":
            print("/reset clears history; /exit closes the model.", flush=True)
            continue

        messages.append({"role": "user", "content": prompt})
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not args.no_thinking,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(args.device)
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "pad_token_id": tokenizer.eos_token_id,
        }
        worker = Thread(target=model.generate, kwargs=generation, daemon=True)
        worker.start()
        print("Qwen> ", end="", flush=True)
        chunks: list[str] = []
        try:
            for chunk in streamer:
                chunks.append(chunk)
                print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            print("\nGeneration interruption requested; waiting for the current decode.", flush=True)
        worker.join()
        response = "".join(chunks).strip()
        print(flush=True)
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    sys.exit(main())
