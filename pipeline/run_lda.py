#!/usr/bin/env python3
"""Parte 5a: LDA (modelagem de tópicos), Python (sklearn) x R (topicmodels).

A LDA é estocástica: Python e R usam algoritmos distintos (online/batch vs VEM),
então não há concordância exata. O cruzamento mede a sobreposição das palavras
de cada tópico (Jaccard do top-N), validando que ambos recuperam tópicos similares.

Uso:
    python run_lda.py --prepared OUT_DIR --k 3 [--level uci] [--n-terms 10]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labiialex_pipeline import Lexique, import_directory, segment_uces  # noqa: E402
from labiialex_pipeline.matrix import build_count_matrix  # noqa: E402
from labiialex_pipeline.preprocess import load_preprocessor  # noqa: E402

R_DIR = Path(__file__).resolve().parent / "r"


def load_prepared(prepared: Path) -> dict:
    return json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="LDA (Python x R topicmodels)")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--k", type=int, default=3, help="número de tópicos")
    parser.add_argument("--level", choices=["uci", "uce"], default="uci")
    parser.add_argument("--n-terms", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stability", action="store_true",
                        help="mede estabilidade dos tópicos entre várias sementes")
    parser.add_argument("--ktuning", action="store_true",
                        help="escolhe k via ldatuning (Griffiths/CaoJuan/Arun/Deveaud)")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    args = parser.parse_args()

    prepared = Path(args.prepared)
    out_dir = prepared / "lda"
    out_dir.mkdir(parents=True, exist_ok=True)
    info = load_prepared(prepared)

    forms = pd.read_csv(prepared / "forms.csv", sep=";", encoding="utf-8")["forme"].astype(str).tolist()
    lexique = Lexique.load(Path(info["dictionaries"]), lang=info.get("lang", "pt"))
    processor = load_preprocessor(info, lexique)
    ucis = import_directory(info["corpus_dir"], drop_speakers=info.get("turn_filter"))
    all_uces = []
    for uci in ucis:
        all_uces.extend(segment_uces(uci, processor, uce_size=info.get("uce_size_target", 40)))
    counts, doc_ids = build_count_matrix(all_uces, forms, level=args.level)
    keep_doc = counts.sum(axis=1) > 0
    counts, doc_ids = counts[keep_doc], [d for d, k in zip(doc_ids, keep_doc) if k]
    print(f"LDA: {counts.shape[0]} documentos ({args.level}) x {counts.shape[1]} formas, k={args.k}.")

    # ----- Python: sklearn -----
    from sklearn.decomposition import LatentDirichletAllocation
    lda = LatentDirichletAllocation(
        n_components=args.k, random_state=args.seed, learning_method="batch", max_iter=50,
    )
    doc_topic = lda.fit_transform(counts)
    topic_term = lda.components_
    py_topics: dict[int, list[str]] = {}
    rows = []
    for t in range(args.k):
        top_idx = np.argsort(topic_term[t])[::-1][: args.n_terms]
        words = [forms[i] for i in top_idx]
        py_topics[t] = words
        for rank, i in enumerate(top_idx, 1):
            rows.append({"topico": t + 1, "rank": rank, "forma": forms[i],
                         "peso": round(float(topic_term[t][i]), 3)})
    pd.DataFrame(rows).to_csv(out_dir / "lda_topics_py.csv", sep=";", index=False, encoding="utf-8")
    dominant = doc_topic.argmax(axis=1) + 1
    pd.DataFrame({"doc_id": doc_ids, "topico_dominante": dominant}).to_csv(
        out_dir / "lda_doc_topics.csv", sep=";", index=False, encoding="utf-8")

    # ----- R: topicmodels -----
    counts_path = out_dir / "_counts.csv"
    pd.DataFrame(counts, index=doc_ids, columns=forms).to_csv(counts_path, sep=";", encoding="utf-8")
    r_out = out_dir / "lda_topics_r.csv"
    proc = subprocess.run(
        ["Rscript", str(R_DIR / "lda_reference.R"), str(counts_path), str(args.k),
         str(r_out), str(args.seed + 1), str(args.n_terms)],
        capture_output=True, text=True, encoding="utf-8")
    print(proc.stdout.strip() or proc.stderr.strip()[:300])

    comparison: dict = {"k": args.k, "level": args.level, "n_docs": int(counts.shape[0])}
    print("\nTópicos (Python / sklearn):")
    for t in range(args.k):
        print(f"  Tópico {t+1}: {', '.join(py_topics[t])}")

    if r_out.exists():
        r_df = pd.read_csv(r_out, sep=";")
        r_topics = {t: r_df[r_df["topico"] == t]["forma"].astype(str).tolist()
                    for t in sorted(r_df["topico"].unique())}
        print("\nTópicos (R / topicmodels):")
        for t, words in r_topics.items():
            print(f"  Tópico {t}: {', '.join(words)}")
        # alinha cada tópico Python ao tópico R de maior sobreposição (Jaccard)
        jaccards = []
        for pt_words in py_topics.values():
            best = 0.0
            sp = set(pt_words)
            for r_words in r_topics.values():
                sr = set(r_words)
                j = len(sp & sr) / len(sp | sr) if (sp | sr) else 0.0
                best = max(best, j)
            jaccards.append(best)
        comparison["mean_topic_top_terms_jaccard"] = round(float(np.mean(jaccards)), 4)
        print(f"\n>>> Sobreposição média das palavras por tópico (Python x R): "
              f"{comparison['mean_topic_top_terms_jaccard']} "
              f"(LDA é estocástica; ~1.0 indica tópicos equivalentes)")
    if args.stability:
        print("\n[estabilidade] Repetindo a LDA com várias sementes ...")
        ref = {t: set(py_topics[t]) for t in py_topics}
        jacc = []
        for s in range(1, 6):
            other = LatentDirichletAllocation(
                n_components=args.k, random_state=args.seed + s,
                learning_method="batch", max_iter=50).fit(counts).components_
            other_topics = [set(forms[i] for i in np.argsort(other[t])[::-1][: args.n_terms])
                            for t in range(args.k)]
            for rt in ref.values():           # alinha cada tópico de referência ao melhor
                best = max((len(rt & ot) / len(rt | ot)) for ot in other_topics)
                jacc.append(best)
        comparison["topic_stability_jaccard"] = round(float(np.mean(jacc)), 4)
        print(f">>> Estabilidade dos tópicos entre sementes: "
              f"{comparison['topic_stability_jaccard']} (perto de 1 = tópicos reprodutíveis)")

    if args.ktuning:
        print(f"\n[k-tuning] ldatuning (Gibbs), k={args.k_min}..{args.k_max} ...")
        kt_out = out_dir / "lda_ktuning.csv"
        kp = subprocess.run(
            ["Rscript", str(R_DIR / "lda_ktuning.R"), str(counts_path),
             str(args.k_min), str(args.k_max), str(kt_out), str(args.seed + 1), "Gibbs"],
            capture_output=True, text=True, encoding="utf-8")
        print(kp.stdout.strip() or kp.stderr.strip()[:300])
        if kt_out.exists():
            kt = pd.read_csv(kt_out, sep=";").sort_values("topics")
            maximize = ["Griffiths2004", "Deveaud2014"]   # maior = melhor
            minimize = ["CaoJuan2009", "Arun2010"]        # menor = melhor
            norm = pd.DataFrame({"k": kt["topics"].astype(int)})
            for col in maximize + minimize:
                v = kt[col].to_numpy(dtype=float)
                rng = np.nanmax(v) - np.nanmin(v)
                z = (v - np.nanmin(v)) / rng if rng > 0 else np.zeros_like(v)
                norm[col] = z if col in maximize else 1.0 - z
            norm["suporte_medio"] = norm[maximize + minimize].mean(axis=1)
            best_k = int(norm.loc[norm["suporte_medio"].idxmax(), "k"])
            per_metric = {}
            for col in maximize:
                per_metric[col] = int(kt.loc[kt[col].idxmax(), "topics"])
            for col in minimize:
                per_metric[col] = int(kt.loc[kt[col].idxmin(), "topics"])
            comparison["ktuning"] = {"k_min": args.k_min, "k_max": args.k_max,
                                     "k_otimo_por_metrica": per_metric,
                                     "k_sugerido_combinado": best_k}
            print("  k ótimo por métrica:", per_metric)
            print(f"  >>> k sugerido (suporte combinado das 4 métricas): {best_k}")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for col in maximize + minimize:
                ax.plot(norm["k"], norm[col], "o-", label=col)
            ax.plot(norm["k"], norm["suporte_medio"], "k--", lw=2, label="suporte médio")
            ax.axvline(best_k, color="green", ls=":", label=f"k sugerido = {best_k}")
            ax.set_xlabel("número de tópicos (k)")
            ax.set_ylabel("métrica normalizada (maior = melhor)")
            ax.set_title("Escolha de k da LDA (ldatuning, 4 métricas)")
            ax.legend(fontsize=8); fig.tight_layout()
            fig.savefig(out_dir / "lda_ktuning.png", dpi=130); plt.close(fig)

    (out_dir / "lda_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados em {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
