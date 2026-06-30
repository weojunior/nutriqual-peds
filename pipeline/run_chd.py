#!/usr/bin/env python3
"""Parte 2: CHD de Reinert com cruzamento Python x R canônico (IRaMuTeQ).

Uso:
    python run_chd.py --prepared OUT_DIR_DA_PARTE1 --n-classes 3 \
        [--rscripts EXTRACTED/internal/Rscripts] [--min-class-size 5]

Lê dtm.csv/uce_meta.csv da Parte 1, roda a CHD em Python (motor canônico limpo)
e roda o CHD.R original do IRaMuTeQ como referência, comparando as partições.

Saídas em OUT_DIR/chd/:
    chd_classes.csv          uce_id, uci_id, variáveis, classe_py, classe_r
    characteristic_terms.csv classe, forma, qui2, sinal (motor Python)
    comparison.json          concordância Python x R (Rand ajustado) e resumo
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labiialex_pipeline.reinert import chd_stability, run_chd  # noqa: E402
from labiialex_pipeline.dendrogram import render_dendrogram  # noqa: E402

DEFAULT_RSCRIPTS = (
    Path(__file__).resolve().parents[1] / "extracted" / "internal" / "Rscripts"
)
R_DRIVER = Path(__file__).resolve().parent / "r" / "chd_reference.R"


def load_dtm(dtm_path: Path) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    """Carrega dtm.csv (uce_id;forma1;...) como matriz esparsa binária."""
    frame = pd.read_csv(dtm_path, sep=";", dtype=str, encoding="utf-8")
    uce_ids = frame.iloc[:, 0].astype(str).tolist()
    forms = list(frame.columns[1:])
    values = frame.iloc[:, 1:].to_numpy(dtype=np.int8)
    return sparse.csr_matrix(values), forms, uce_ids


def run_r_reference(
    rscripts_dir: Path, dtm_path: Path, n_classes: int, out_dir: Path
) -> dict[str, str] | None:
    """Roda o CHD.R canônico; retorna {uce_id: classe} ou None se falhar."""
    out_csv = out_dir / "chd_classes_r.csv"
    log = out_dir / "chd_reference_R.log"
    proc = subprocess.run(
        [
            "Rscript", str(R_DRIVER), str(rscripts_dir), str(dtm_path),
            str(n_classes), str(out_csv), str(log),
        ],
        capture_output=True, text=True, encoding="utf-8",
    )
    print(proc.stdout.strip())
    if proc.returncode != 0 or not out_csv.exists():
        print(f"[aviso] referência R indisponível (rc={proc.returncode}); "
              f"seguindo só com Python. Log: {log}")
        if proc.stderr.strip():
            print(proc.stderr.strip()[:500])
        return None
    r_frame = pd.read_csv(out_csv, sep=";", dtype=str, encoding="utf-8")
    return dict(zip(r_frame["uce_id"], r_frame["classe"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="CHD de Reinert (Python x R)")
    parser.add_argument("--prepared", required=True, help="diretório da Parte 1")
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--min-class-size", type=int, default=5)
    parser.add_argument("--no-relocate", action="store_true",
                        help="desliga a fase de realocação (mais rápido, menos fiel)")
    parser.add_argument("--stability", action="store_true",
                        help="estima estabilidade das classes por subamostragem")
    parser.add_argument("--rscripts", default=str(DEFAULT_RSCRIPTS))
    args = parser.parse_args()

    prepared = Path(args.prepared)
    out_dir = prepared / "chd"
    out_dir.mkdir(parents=True, exist_ok=True)

    dtm_path = prepared / "dtm.csv"
    dtm, forms, uce_ids = load_dtm(dtm_path)
    print(f"Matriz: {dtm.shape[0]} UCEs x {dtm.shape[1]} formas ativas.")

    print(f"[Python] CHD de Reinert (alvo {args.n_classes} classes"
          f"{'' if not args.no_relocate else ', sem realocação'}) ...")
    result = run_chd(
        dtm, forms, uce_ids,
        n_classes=args.n_classes, min_class_size=args.min_class_size,
        relocate=not args.no_relocate,
    )
    print(f"[Python] {result.n_classes} classes: {result.class_sizes}")

    print("[R] CHD.R canônico do IRaMuTeQ (referência) ...")
    r_map = run_r_reference(Path(args.rscripts), dtm_path, args.n_classes, out_dir)

    # ----- tabela de classes (join com metadados) -----
    meta_path = prepared / "uce_meta.csv"
    meta = pd.read_csv(meta_path, sep=";", dtype=str, encoding="utf-8")
    py_map = {uid: int(c) for uid, c in zip(result.uce_ids, result.assignments)}
    meta = meta[meta["uce_id"].isin(py_map)].copy()
    meta["classe_py"] = meta["uce_id"].map(py_map)
    if r_map is not None:
        meta["classe_r"] = meta["uce_id"].map(lambda u: int(r_map.get(u, 0)))
    meta.to_csv(out_dir / "chd_classes.csv", sep=";", index=False, encoding="utf-8")

    # ----- formas características (Python) -----
    rows = []
    for class_id, terms in result.characteristic_terms.items():
        for term in terms:
            rows.append(
                {"classe": class_id, "forma": term.form,
                 "qui2": round(term.chi2, 3), "sinal": term.sign,
                 "n_classe": term.n_in_class, "n_total": term.n_total}
            )
    pd.DataFrame(rows).to_csv(
        out_dir / "characteristic_terms.csv", sep=";", index=False, encoding="utf-8"
    )

    # ----- segmentos típicos por classe (trechos reais para interpretar) -----
    texts_path = prepared / "uce_texts.csv"
    if texts_path.exists():
        texts = pd.read_csv(texts_path, sep=";", dtype=str, encoding="utf-8")
        text_map = dict(zip(texts["uce_id"], texts["texto"].fillna("")))
        dense = dtm.toarray()
        form_idx = {f: j for j, f in enumerate(forms)}
        seg_rows = []
        for class_id, terms in result.characteristic_terms.items():
            weight = {t.form: t.chi2 for t in terms if t.sign == "+"}
            cols = [(form_idx[f], w) for f, w in weight.items() if f in form_idx]
            members = [i for i, c in enumerate(result.assignments) if c == class_id]
            scored = []
            for i in members:
                score = sum(w for j, w in cols if dense[i, j] > 0)
                scored.append((score, result.uce_ids[i]))
            scored.sort(reverse=True)
            for rank, (score, uid) in enumerate(scored[:5], 1):
                seg_rows.append({"classe": class_id, "rank": rank, "uce_id": uid,
                                 "score": round(float(score), 2),
                                 "texto": text_map.get(uid, "")})
        pd.DataFrame(seg_rows).to_csv(
            out_dir / "typical_segments.csv", sep=";", index=False, encoding="utf-8")
        print("\nSegmentos típicos por classe (motor Python, top 2):")
        seg_df = pd.DataFrame(seg_rows)
        for class_id in sorted(result.characteristic_terms):
            top2 = seg_df[(seg_df["classe"] == class_id) & (seg_df["rank"] <= 2)]
            for _, r in top2.iterrows():
                print(f"  C{class_id} ({r['score']}): {r['texto'][:90]}")

    # ----- dendrograma da árvore de divisões -----
    try:
        dendro_path = render_dendrogram(result, out_dir / "dendrogram.png")
        print(f"\nDendrograma -> {dendro_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[aviso] dendrograma não gerado: {exc}")

    # ----- comparação Python x R -----
    # ----- avisos estatísticos -----
    warnings: list[str] = []
    if result.n_classes < args.n_classes:
        warnings.append(f"Atingiu {result.n_classes} de {args.n_classes} classes pedidas "
                        "(as demais não eram divisíveis com o tamanho mínimo).")
    small = [c for c, s in result.class_sizes.items() if s < 2 * args.min_class_size]
    if small:
        warnings.append(f"Classes pequenas (<{2*args.min_class_size} UCEs): {small}.")
    fragile = {
        c: sum(1 for t in terms if t.min_expected < 5)
        for c, terms in result.characteristic_terms.items()
    }
    total_fragile = sum(fragile.values())
    if total_fragile:
        warnings.append(f"{total_fragile} formas características têm célula esperada <5 "
                        "(χ² frágil; interprete com cautela). Por classe: "
                        + ", ".join(f"C{c}:{n}" for c, n in fragile.items() if n))
    for w in warnings:
        print(f"  [aviso] {w}")

    comparison: dict = {
        "n_uce": int(dtm.shape[0]),
        "n_forms": int(dtm.shape[1]),
        "python": {"n_classes": result.n_classes, "class_sizes": result.class_sizes},
        "warnings": warnings,
    }
    if r_map is not None:
        from sklearn.metrics import adjusted_rand_score
        common = meta.dropna(subset=["classe_py", "classe_r"])
        common = common[common["classe_r"] > 0]
        ari = adjusted_rand_score(common["classe_py"], common["classe_r"])
        comparison["r_reference"] = {
            "n_classes": int(common["classe_r"].nunique()),
            "adjusted_rand_index_vs_python": round(float(ari), 4),
            "n_compared": int(len(common)),
        }
        print(f"\n>>> Concordância Python x R (Rand ajustado): {ari:.4f} "
              f"(1.0 = partições idênticas)")
    else:
        comparison["r_reference"] = None

    if args.stability:
        print("[Python] Estimando estabilidade das classes por subamostragem ...")
        stab = chd_stability(dtm, forms, uce_ids, result.assignments,
                             n_classes=args.n_classes, min_class_size=args.min_class_size)
        comparison["stability"] = stab
        print(f">>> Estabilidade: within={stab['within_class_coassoc']} "
              f"between={stab['between_class_coassoc']} (índice {stab['stability']}; "
              "perto de 1 = classes muito estáveis)")

    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResultados em {out_dir}/")

    # imprime as formas características mais fortes por classe (Python)
    print("\nFormas características por classe (motor Python, top 8):")
    for class_id in sorted(result.characteristic_terms):
        top = [t.form for t in result.characteristic_terms[class_id] if t.sign == "+"][:8]
        print(f"  Classe {class_id} (n={result.class_sizes[class_id]}): {', '.join(top)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
