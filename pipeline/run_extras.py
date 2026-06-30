#!/usr/bin/env python3
"""Parte 9: análises complementares (nuvem, n-gramas, árvore de palavras, heatmap, YAKE).

Uso:
    python run_extras.py --prepared OUT_DIR --what wordcloud
    python run_extras.py --prepared OUT_DIR --what ngrams --n 2
    python run_extras.py --prepared OUT_DIR --what wordtree --query criança --direction right
    python run_extras.py --prepared OUT_DIR --what heatmap        (requer Parte 2)
    python run_extras.py --prepared OUT_DIR --what yake --n 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labiialex_pipeline import Lexique, import_directory, segment_uces  # noqa: E402
from labiialex_pipeline import extras  # noqa: E402
from labiialex_pipeline.preprocess import load_preprocessor  # noqa: E402
from labiialex_pipeline.tokenize import tokenize  # noqa: E402


def _load(prepared: Path):
    info = json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))
    lexique = Lexique.load(Path(info["dictionaries"]), lang=info.get("lang", "pt"))
    processor = load_preprocessor(info, lexique)
    ucis = import_directory(info["corpus_dir"], drop_speakers=info.get("turn_filter"))
    return info, processor, ucis


def _segment(ucis, processor, info):
    out = []
    for uci in ucis:
        out.extend(segment_uces(uci, processor, uce_size=info.get("uce_size_target", 40)))
    return out


def do_wordcloud(prepared, out_dir):
    from wordcloud import WordCloud
    forms = pd.read_csv(prepared / "forms.csv", sep=";", encoding="utf-8")
    freq = dict(zip(forms["forme"].astype(str), forms["frequencia"].astype(int)))
    wc = WordCloud(width=1100, height=750, background_color="white",
                   colormap="viridis").generate_from_frequencies(freq)
    wc.to_file(str(out_dir / "wordcloud.png"))
    print(f"Nuvem de palavras -> {out_dir / 'wordcloud.png'} ({len(freq)} formas)")


def do_ngrams(prepared, out_dir, lexique, ucis, info, n):
    uces = _segment(ucis, lexique, info)
    counts = extras.ngram_counts(uces, n=n)
    rows = [{"ngrama": " ".join(g), "frequencia": c} for g, c in counts.most_common(200)]
    label = {2: "bigramas", 3: "trigramas"}.get(n, f"{n}-gramas")
    path = out_dir / f"{label}.csv"
    pd.DataFrame(rows).to_csv(path, sep=";", index=False, encoding="utf-8")
    print(f"{label.capitalize()} -> {path}")
    for r in rows[:12]:
        print(f"  {r['frequencia']:4d}  {r['ngrama']}")


def do_wordtree(prepared, out_dir, lexique, ucis, query, direction, depth):
    tree = extras.word_tree(ucis, lexique, query, direction=direction, depth=depth)
    lines = extras.render_word_tree(tree)
    safe = re.sub(r"[^\w]+", "_", query)
    path = out_dir / f"wordtree_{safe}_{direction}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Árvore de palavras ('{query}', {direction}) -> {path}\n")
    print("\n".join(lines[:25]))


def do_heatmap(prepared, out_dir, top):
    chd = prepared / "chd" / "chd_classes.csv"
    if not chd.exists():
        print("Heatmap requer a Parte 2 (run_chd.py)."); return
    frame = pd.read_csv(prepared / "dtm.csv", sep=";", dtype=str, encoding="utf-8")
    forms = list(frame.columns[1:])
    dtm = frame.iloc[:, 1:].to_numpy(dtype=float)
    uce_ids = frame.iloc[:, 0].astype(str).tolist()
    meta = pd.read_csv(chd, sep=";", dtype=str, encoding="utf-8").set_index("uce_id")
    assignments = np.array([int(meta.loc[u, "classe_py"]) for u in uce_ids])
    n_classes = int(assignments.max())
    mat, rows, cols = extras.form_class_proportions(dtm, assignments, forms, n_classes, top=top)
    fig, ax = plt.subplots(figsize=(1.4 * n_classes + 2, 0.32 * len(rows) + 1))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    ax.set_title("Proporção de presença da forma por classe")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout(); fig.savefig(out_dir / "heatmap_forms_classes.png", dpi=130)
    plt.close(fig)
    print(f"Heatmap -> {out_dir / 'heatmap_forms_classes.png'}")


def do_yake(prepared, out_dir, processor, ucis, info, n):
    text = "\n".join(u.text for u in ucis)
    # mapeia lema -> superfície: toda forma não ativa (suplementar ou stopword,
    # em qualquer flexão presente no corpus) vira stopword do YAKE.
    surface_stops = {tok for tok in set(tokenize(text)) if not processor.is_active(tok)}
    kws = extras.yake_keyphrases(text, lang=info.get("lang", "pt"), n=n, top=30,
                                 stopwords=surface_stops)
    rows = [{"expressao": k, "score_yake": round(s, 4)} for k, s in kws]
    path = out_dir / "yake_keyphrases.csv"
    pd.DataFrame(rows).to_csv(path, sep=";", index=False, encoding="utf-8")
    print(f"YAKE (menor score = mais relevante) -> {path}")
    for r in rows[:15]:
        print(f"  {r['score_yake']:.4f}  {r['expressao']}")


def main() -> int:
    p = argparse.ArgumentParser(description="Análises complementares")
    p.add_argument("--prepared", required=True)
    p.add_argument("--what", required=True,
                   choices=["wordcloud", "ngrams", "wordtree", "heatmap", "yake"])
    p.add_argument("--n", type=int, default=2, help="n para n-gramas/YAKE")
    p.add_argument("--query", default="", help="palavra-chave (wordtree)")
    p.add_argument("--direction", choices=["right", "left"], default="right")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--top", type=int, default=30, help="formas no heatmap")
    args = p.parse_args()

    prepared = Path(args.prepared)
    out_dir = prepared / "extras"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.what == "wordcloud":
        do_wordcloud(prepared, out_dir)
    elif args.what == "heatmap":
        do_heatmap(prepared, out_dir, args.top)
    else:
        info, lexique, ucis = _load(prepared)
        if args.what == "ngrams":
            do_ngrams(prepared, out_dir, lexique, ucis, info, args.n)
        elif args.what == "wordtree":
            if not args.query:
                print("Informe --query para a árvore de palavras."); return 2
            do_wordtree(prepared, out_dir, lexique, ucis, args.query, args.direction, args.depth)
        elif args.what == "yake":
            do_yake(prepared, out_dir, lexique, ucis, info, args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
