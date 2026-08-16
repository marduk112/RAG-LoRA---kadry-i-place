"""Skrypty w scripts/ nie są pakietem (celowo -- to zbiór niezależnych
CLI, patrz README) i importują się nawzajem po nazwie modułu, np.
`from prompt import ...`. Testy potrzebują tego samego na sys.path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
