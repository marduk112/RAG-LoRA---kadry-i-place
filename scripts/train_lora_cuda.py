"""
UWAGA: NIEPRZETESTOWANE. Ten skrypt został napisany bez dostępu do
karty NVIDIA/CUDA (środowisko deweloperskie to Mac/Apple Silicon) --
API (peft, trl, transformers) zweryfikowano względem aktualnej
dokumentacji, ale end-to-end przebieg treningu nie został uruchomiony.
Jeśli coś nie zadziała, to najpewniej drobna niezgodność wersji
bibliotek -- daj znać (issue) albo od razu wyślij PR z poprawką i
aktualizacją PROGRESS.md.

Odpowiednik `mlx_lm.lora` (patrz PROGRESS.md, kroki 4b i 6) dla
Linuksa/Windows z kartą NVIDIA: LoRA fine-tuning Bielika przez
transformers + peft + bitsandbytes (kwantyzacja 4-bit) + trl
(SFTTrainer). Trenuje na tych samych plikach co wariant MLX
(data/finetune/train.jsonl, valid.jsonl, format {"messages": [...]}).

Domyślne hiperparametry (batch 2, 200 kroków, ewaluacja/zapis co 25,
lr 1e-5, maskowanie promptu w loss) są 1:1 przeniesione z przebiegów
na MLX opisanych w PROGRESS.md. Tam model przeuczał się bardzo szybko
(najlepszy val loss już przy iteracji 25-50 z 200) -- to zjawisko
zależy od optymalizatora/precyzji i może wypaść inaczej tutaj, więc
mimo wspólnego punktu startowego OBSERWUJ `eval_loss` w logach i wybierz
checkpoint z najniższą wartością, a nie automatycznie ostatni
(patrz --save-steps -- każdy zapis trafia do osobnego podkatalogu
checkpoint-N w --output-dir).

Użycie:
    python scripts/train_lora_cuda.py

    python scripts/train_lora_cuda.py \\
        --model speakleash/Bielik-4.5B-v3.0-Instruct \\
        --output-dir adapters-cuda/bielik4.5b-kadry-lora
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "finetune"
DEFAULT_MODEL = "speakleash/Bielik-11B-v3.0-Instruct"
DEFAULT_OUTPUT_DIR = ROOT / "adapters-cuda" / "bielik11b-kadry-lora"


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning Bielika na CUDA (transformers + peft + trl)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Repo Hugging Face modelu bazowego")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Katalog z train.jsonl / valid.jsonl")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[INFO] Ładowanie modelu bazowego ({args.model}) w 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Bielik's stock chat template (ChatML: <|im_start|>role\n...<|im_end|>\n) ma
    # role/content sklejone w jednym {{ }} bloku, więc trl nie potrafi go
    # automatycznie załatać markerami {% generation %} wymaganymi przez
    # assistant_only_loss (patrz ValueError "chat template is not
    # training-compatible"). Wersja niżej jest funkcjonalnie identyczna --
    # renderuje ten sam tekst -- tylko z turą asystenta owiniętą w
    # {% generation %}, żeby loss liczył się wyłącznie na jej tokenach.
    tokenizer.chat_template = (
        "{{bos_token}}{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% generation %}{{message['content'] + '<|im_end|>' + '\n'}}{% endgeneration %}"
        "{% else %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],  # jak w domyślnej konfiguracji mlx_lm.lora (tylko attention)
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"[INFO] Wczytywanie danych z {data_dir}...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "valid.jsonl"),
        },
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=10,
        assistant_only_loss=True,  # odpowiednik --mask-prompt w mlx_lm.lora
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
    trainer.train()

    final_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_path))
    print(f"[INFO] Zapisano finalny adapter w {final_path}")
    print(
        "[INFO] Sprawdź eval_loss w logach powyżej i porównaj z checkpointami w "
        f"{args.output_dir}/checkpoint-* -- w naszych przebiegach na MLX najlepszy "
        "checkpoint NIE był ostatnim (patrz PROGRESS.md)."
    )


if __name__ == "__main__":
    main()
