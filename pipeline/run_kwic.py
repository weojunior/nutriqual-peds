#!/usr/bin/env python3
"""Parte 5b: concordâncias KWIC (keyword in context).

Uso:
    python run_kwic.py --prepared OUT_DIR --query alimentação [--mode lemma] [--window 6]

Saída: OUT_DIR/kwic/kwic_<query>.csv  e amostra no terminal.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labiialex_pipeline import Lexique, import_directory  # noqa: E402
from labiialex_pipeline.kwic import concordance  # noqa: E402
from labiialex_pipeline.preprocess import load_preprocessor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Concordâncias KWIC")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=["lemma", "exact"], default="lemma")
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--max", type=int, default=40, help="linhas mostradas no terminal")
    args = parser.parse_args()

    prepared = Path(args.prepared)
    info = json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))
    lexique = Lexique.load(Path(info["dictionaries"]), lang=info.get("lang", "pt"))
    processor = load_preprocessor(info, lexique)
    ucis = import_directory(info["corpus_dir"], drop_speakers=info.get("turn_filter"))

    lines = concordance(ucis, processor, args.query, window=args.window, mode=args.mode)
    out_dir = prepared / "kwic"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w]+", "_", args.query)
    rows = [{"uci_id": kl.uci_id, **kl.variables, "esquerda": kl.left,
             "palavra": kl.keyword, "direita": kl.right} for kl in lines]
    out_path = out_dir / f"kwic_{safe}.csv"
    pd.DataFrame(rows).to_csv(out_path, sep=";", index=False, encoding="utf-8")

    print(f"'{args.query}' (modo {args.mode}): {len(lines)} ocorrências em "
          f"{len({kl.uci_id for kl in lines})} documentos.")
    print(f"Concordância completa em {out_path}\n")
    for kl in lines[: args.max]:
        print(f"  [{kl.uci_id}] ...{kl.left:>40s} [ {kl.keyword} ] {kl.right}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
