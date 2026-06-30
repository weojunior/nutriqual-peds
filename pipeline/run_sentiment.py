#!/usr/bin/env python3
"""Parte 5c: sentimento pt-BR (léxico + negação). Baseline transparente, não clínico.

Uso:
    python run_sentiment.py --prepared OUT_DIR [--level uci] [--lexicon meu_lexico.csv]

--lexicon (opcional): CSV com colunas 'forma,polaridade' (polaridade = pos|neg)
para ampliar o léxico-semente com termos do seu domínio.

Saída: OUT_DIR/sentiment/sentiment_<level>.csv + resumo por variável.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labiialex_pipeline import Lexique, import_directory, segment_uces  # noqa: E402
from labiialex_pipeline.preprocess import load_preprocessor  # noqa: E402
from labiialex_pipeline.sentiment import (  # noqa: E402
    SEED_NEGATIVE,
    SEED_POSITIVE,
    label_for,
    score_text,
)


def load_extra_lexicon(path: Path, processor):
    pos, neg = set(SEED_POSITIVE), set(SEED_NEGATIVE)
    frame = pd.read_csv(path, dtype=str).fillna("")
    for _, row in frame.iterrows():
        lemma = processor.lemma(str(row["forma"]).lower().strip())
        if str(row["polaridade"]).lower().startswith("p"):
            pos.add(lemma)
        elif str(row["polaridade"]).lower().startswith("n"):
            neg.add(lemma)
    return frozenset(pos), frozenset(neg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sentimento pt-BR (léxico + negação)")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--level", choices=["uci", "uce"], default="uci")
    parser.add_argument("--lexicon", default="")
    args = parser.parse_args()

    prepared = Path(args.prepared)
    info = json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))
    lexique = Lexique.load(Path(info["dictionaries"]), lang=info.get("lang", "pt"))
    processor = load_preprocessor(info, lexique)
    ucis = import_directory(info["corpus_dir"], drop_speakers=info.get("turn_filter"))

    positive, negative = SEED_POSITIVE, SEED_NEGATIVE
    if args.lexicon:
        positive, negative = load_extra_lexicon(Path(args.lexicon), processor)
    print(f"Léxico: {len(positive)} positivos, {len(negative)} negativos.")

    rows = []
    if args.level == "uci":
        for uci in ucis:
            score, npos, nneg = score_text(uci.text, processor, positive, negative)
            rows.append({"doc_id": uci.uci_id, **uci.variables,
                         "score": score, "n_pos": npos, "n_neg": nneg,
                         "rotulo": label_for(score)})
    else:
        for uci in ucis:
            for uce in segment_uces(uci, processor, uce_size=info.get("uce_size_target", 40)):
                score, npos, nneg = score_text(uce.text, processor, positive, negative)
                rows.append({"doc_id": uce.uce_id, "uci_id": uci.uci_id, **uci.variables,
                             "score": score, "n_pos": npos, "n_neg": nneg,
                             "rotulo": label_for(score)})

    frame = pd.DataFrame(rows)
    out_dir = prepared / "sentiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sentiment_{args.level}.csv"
    frame.to_csv(out_path, sep=";", index=False, encoding="utf-8")

    print(f"\nDistribuição geral: "
          f"{dict(frame['rotulo'].value_counts())}")
    var_cols = [c for c in frame.columns
                if c not in {"doc_id", "uci_id", "score", "n_pos", "n_neg", "rotulo"}]
    for var in var_cols:
        if frame[var].nunique() > 1:
            print(f"\nScore médio por '{var}':")
            print(frame.groupby(var)["score"].mean().round(2).to_string())
    print(f"\nResultados em {out_path}")
    print("Aviso: baseline lexical, não validado clinicamente; amplie o léxico ao seu domínio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
