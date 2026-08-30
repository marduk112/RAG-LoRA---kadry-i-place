#!/bin/bash
# Ręczne testy end-to-end sprawdzające, czy model przyznaje się do
# niewiedzy zamiast zgadywać liczby spoza kontekstu RAG (patrz
# PROGRESS.md, Krok 26) -- val loss nie jest tu wiarygodnym
# predyktorem, trzeba sprawdzać ręcznie na docelowych pytaniach.
# Odpowiednik scripts/test_hedge_probes_mlx.sh dla wariantu CUDA --
# porównuje adapter bielik11b-kadry-lora-hedge z modelem bazowym bez
# adaptera na 4 pytaniach: jeden fakt z RAG (PPK), jeden temat trafiony
# ale bez konkretnej liczby w RAG (dieta), jeden całkiem spoza domeny
# (VAT), jeden graniczny przypadek stażu pracy.
#
# Wymaga karty NVIDIA i wytrenowanego adaptera pod ścieżką niżej
# (patrz PROGRESS.md, Krok 26 -- train_lora_cuda.py).
#
# Użycie: ./scripts/test_hedge_probes_cuda.sh (z katalogu głównego repo)

set -e
cd "$(dirname "$0")/.."

MODEL="speakleash/Bielik-11B-v3.0-Instruct"
ADAPTER="adapters-cuda/bielik11b-kadry-lora-hedge/final"

run() {
    local label="$1"
    local adapter_flag="$2"
    local prompt="$3"
    echo ""
    echo "=================================================================="
    echo "### $label"
    echo "=================================================================="
    .venv/bin/python scripts/chat_cuda.py --model "$MODEL" $adapter_flag --prompt "$prompt"
}

run "1. PPK -- z adapterem" \
    "--adapter-path $ADAPTER" \
    "Ile procent wynagrodzenia wpłaca pracodawca do PPK jako wpłatę podstawową?"

run "2. Dieta zagraniczna Niemcy -- z adapterem" \
    "--adapter-path $ADAPTER" \
    "Jaka jest wysokość diety za dobę zagranicznej podróży służbowej do Niemiec?"

run "2b. Dieta zagraniczna Niemcy -- BEZ adaptera" \
    "--no-adapter" \
    "Jaka jest wysokość diety za dobę zagranicznej podróży służbowej do Niemiec?"

run "3. VAT -- z adapterem" \
    "--adapter-path $ADAPTER" \
    "Jaka jest aktualna stawka podatku VAT dla usług księgowych?"

run "3b. VAT -- BEZ adaptera" \
    "--no-adapter" \
    "Jaka jest aktualna stawka podatku VAT dla usług księgowych?"

run "4. Staż dokładnie 3 lata / okres wypowiedzenia -- z adapterem" \
    "--adapter-path $ADAPTER" \
    "Pracownik jest zatrudniony u tego samego pracodawcy dokładnie 3 lata. Jaki obowiązuje okres wypowiedzenia?"

echo ""
echo "Gotowe."
