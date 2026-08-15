"""
UWAGA: NIEPRZETESTOWANE (patrz train_lora_cuda.py -- ten sam zastrzeżenie:
napisane bez dostępu do karty NVIDIA/CUDA, API zweryfikowane względem
dokumentacji, ale nie uruchomione end-to-end).

Odpowiednik scripts/chat.py dla Linuksa/Windows z kartą NVIDIA:
RAG (rag_search.RagIndex) + Bielik doduczony przez
scripts/train_lora_cuda.py, generacja przez transformers + peft
zamiast MLX. Ten sam prompt systemowy i budowanie kontekstu co
pozostałe warianty (scripts/prompt.py).

Użycie:
    python scripts/chat_cuda.py
    python scripts/chat_cuda.py --prompt "pytanie..."
    python scripts/chat_cuda.py --no-adapter
    python scripts/chat_cuda.py \\
        --model speakleash/Bielik-11B-v3.0-Instruct \\
        --adapter-path adapters-cuda/bielik11b-kadry-lora/final
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompt import SYSTEM_PROMPT, build_user_message
from rag_search import RagIndex

DEFAULT_MODEL = "speakleash/Bielik-11B-v3.0-Instruct"
DEFAULT_ADAPTER_PATH = "adapters-cuda/bielik11b-kadry-lora/final"


def load_model(model_id: str, adapter_path: str | None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def answer(model, tokenizer, rag: RagIndex, question: str, top_k: int) -> str:
    results = rag.search(question, top_k=top_k)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, results)},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Asystent kadrowo-płacowy (RAG + Bielik LoRA, CUDA)")
    parser.add_argument("--prompt", type=str, default=None, help="Pojedyncze pytanie zamiast trybu interaktywnego")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Repo Hugging Face modelu bazowego")
    parser.add_argument("--adapter-path", type=str, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--no-adapter", action="store_true", help="Użyj bazowego modelu bez LoRA")
    parser.add_argument("--top-k", type=int, default=5, help="Liczba fragmentów RAG dołączanych do kontekstu")
    args = parser.parse_args()

    adapter_path = None if args.no_adapter else args.adapter_path
    print(f"[INFO] Ładowanie modelu ({args.model}, adapter={adapter_path or '(brak)'})...")
    model, tokenizer = load_model(args.model, adapter_path)

    print("[INFO] Ładowanie indeksu RAG...")
    rag = RagIndex()

    if args.prompt:
        print(answer(model, tokenizer, rag, args.prompt, args.top_k))
        return

    print("\nAsystent kadrowo-płacowy (CUDA) gotowy. Wpisz pytanie (Ctrl+C aby zakończyć).\n")
    while True:
        try:
            question = input("Ty: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        text = answer(model, tokenizer, rag, question, args.top_k)
        print("Asystent:", text, "\n")


if __name__ == "__main__":
    main()
