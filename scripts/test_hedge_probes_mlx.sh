#!/bin/bash
# Ręczne testy end-to-end sprawdzające, czy model przyznaje się do
# niewiedzy zamiast zgadywać liczby spoza kontekstu RAG (patrz
# PROGRESS.md, Krok 26/27) -- val loss nie jest tu wiarygodnym
# predyktorem, trzeba sprawdzać ręcznie na docelowych pytaniach.
# Porównuje checkpointy iter25 vs iter50 adaptera
# bielik11b-kadry-lora-hedge na 4 pytaniach: jeden fakt z RAG (PPK),
# jeden temat trafiony ale bez konkretnej liczby w RAG (dieta), jeden
# całkiem spoza domeny (VAT), jeden graniczny przypadek stażu pracy.
#
# Użycie: ./scripts/test_hedge_probes_mlx.sh (z katalogu głównego repo)

set -e
cd "$(dirname "$0")/.."

run() {
    local label="$1"
    local adapter_flag="$2"
    local prompt="$3"
    echo ""
    echo "=================================================================="
    echo "### $label"
    echo "=================================================================="
    .venv/bin/python scripts/chat.py $adapter_flag --prompt "$prompt"
}

run "1a. PPK -- iter25" "--adapter-path adapters/bielik11b-kadry-lora-hedge-iter25" \
    "Ile procent wynagrodzenia wpłaca pracodawca do PPK jako wpłatę podstawową?"
run "1b. PPK -- iter50" "--adapter-path adapters/bielik11b-kadry-lora-hedge-iter50" \
    "Ile procent wynagrodzenia wpłaca pracodawca do PPK jako wpłatę podstawową?"

run "2a. Dieta Niemcy -- iter25" "--adapter-path adapters/bielik11b-kadry-lora-hedge-iter25" \
    "Jaka jest wysokość diety za dobę zagranicznej podróży służbowej do Niemiec?"
run "2b. Dieta Niemcy -- iter50" "--adapter-path adapters/bielik11b-kadry-lora-hedge-iter50" \
    "Jaka jest wysokość diety za dobę zagranicznej podróży służbowej do Niemiec?"

run "3a. VAT -- iter25" "--adapter-path adapters/bielik11b-kadry-lora-hedge-iter25" \
    "Jaka jest aktualna stawka podatku VAT dla usług księgowych?"
run "3b. VAT -- iter50" "--adapter-path adapters/bielik11b-kadry-lora-hedge-iter50" \
    "Jaka jest aktualna stawka podatku VAT dla usług księgowych?"

run "4a. Staż dokładnie 3 lata -- iter25" "--adapter-path adapters/bielik11b-kadry-lora-hedge-iter25" \
    "Pracownik jest zatrudniony u tego samego pracodawcy dokładnie 3 lata. Jaki obowiązuje okres wypowiedzenia?"
run "4b. Staż dokładnie 3 lata -- iter50" "--adapter-path adapters/bielik11b-kadry-lora-hedge-iter50" \
    "Pracownik jest zatrudniony u tego samego pracodawcy dokładnie 3 lata. Jaki obowiązuje okres wypowiedzenia?"

echo ""
echo "Gotowe."
