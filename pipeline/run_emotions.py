#!/usr/bin/env python3
"""Parte 8: emoções (NRC via syuzhet) por documento e por variável.

Completa o módulo de sentimento com as 8 categorias de emoção de Plutchik/NRC
(raiva, antecipação, nojo, medo, alegria, tristeza, surpresa, confiança) mais
valência (positivo/negativo), usando o mesmo motor (syuzhet) do software original.

Uso:
    python run_emotions.py --prepared OUT_DIR [--variables grupo,tema]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labiialex_pipeline import import_directory  # noqa: E402

R_DIR = Path(__file__).resolve().parent / "r"
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
EMO_PT = {"anger": "raiva", "anticipation": "antecipação", "disgust": "nojo",
          "fear": "medo", "joy": "alegria", "sadness": "tristeza",
          "surprise": "surpresa", "trust": "confiança"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Emoções NRC (syuzhet)")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--variables", default="")
    args = parser.parse_args()

    prepared = Path(args.prepared)
    out_dir = prepared / "emotions"
    out_dir.mkdir(parents=True, exist_ok=True)
    info = json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))
    ucis = import_directory(info["corpus_dir"], drop_speakers=info.get("turn_filter"))

    var_names: list[str] = []
    rows = []
    for uci in ucis:
        rows.append({"doc_id": uci.uci_id, "texto": uci.text.replace("\n", " "),
                     **uci.variables})
        for v in uci.variables:
            if v not in var_names:
                var_names.append(v)
    docs = pd.DataFrame(rows)
    docs_path = out_dir / "_docs.csv"
    docs.to_csv(docs_path, sep=";", index=False, encoding="utf-8")

    print(f"Calculando emoções NRC (syuzhet) em {len(docs)} documentos ...")
    emo_out = out_dir / "emotions_per_doc.csv"
    proc = subprocess.run(
        ["Rscript", str(R_DIR / "emotions_reference.R"), str(docs_path), str(emo_out)],
        capture_output=True, text=True, encoding="utf-8")
    print(proc.stdout.strip() or proc.stderr.strip()[:400])
    if not emo_out.exists():
        print("Falha ao calcular emoções (syuzhet)."); return 3

    emo = pd.read_csv(emo_out, sep=";")
    emo = emo.merge(docs[["doc_id", *var_names]], on="doc_id", how="left")
    emo.rename(columns=EMO_PT).to_csv(out_dir / "emotions_per_doc.csv", sep=";",
                                      index=False, encoding="utf-8")

    # médias gerais e por variável
    overall = emo[EMOTIONS].mean()
    print("\nEmoções médias (corpus):")
    for e in EMOTIONS:
        print(f"  {EMO_PT[e]:12s} {overall[e]:.2f}")

    # média geral no corpus inteiro (gráfico único, sem estratificar)
    ov = overall.rename(index=EMO_PT).sort_values(ascending=False)
    ov.to_frame("media_corpus").to_csv(out_dir / "emotions_overall.csv",
                                       sep=";", encoding="utf-8")
    ax = ov.plot(kind="bar", figsize=(9, 5), color="#4C72B0", legend=False)
    ax.set_title("Emoções médias no corpus (NRC/syuzhet)")
    ax.set_ylabel("intensidade média (ocorrências por documento)")
    ax.set_xlabel("")
    ax.bar_label(ax.containers[0], fmt="%.1f", padding=2, fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(out_dir / "emotions_overall.png", dpi=130)
    plt.close()
    print("\nMédia geral -> emotions_overall.csv / emotions_overall.png")

    variables = [v.strip() for v in args.variables.split(",") if v.strip()] or var_names
    for var in variables:
        if var in emo.columns and emo[var].nunique() > 1:
            grp = emo.groupby(var)[EMOTIONS].mean().rename(columns=EMO_PT)
            grp.to_csv(out_dir / f"emotions_by_{var}.csv", sep=";", encoding="utf-8")
            # gráfico de barras agrupadas
            ax = grp.T.plot(kind="bar", figsize=(11, 6))
            ax.set_title(f"Emoções médias por '{var}' (NRC/syuzhet)")
            ax.set_ylabel("intensidade média")
            plt.tight_layout(); plt.savefig(out_dir / f"emotions_by_{var}.png", dpi=130)
            plt.close()
            print(f"\nEmoções por '{var}' -> emotions_by_{var}.csv / .png")

    print(f"\nResultados em {out_dir}/")
    print("Aviso: NRC é um léxico traduzido automaticamente; trate como exploratório.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
