#!/usr/bin/env python3
"""Análise de saturação lexical incremental (grupo a grupo).

Mede, ao acrescentar cada grupo (ordem cronológica e ordens aleatórias), quando a
estrutura lexical estabiliza, e compara com a saturação temática declarada.

Indicadores por passo k (primeiros k grupos):
  - proporção de formas ativas NOVAS (estabilização do vocabulário relevante);
  - estabilidade da CHD: índice de Rand ajustado entre o passo k-1 e o k, nas UCEs comuns;
  - sobreposição (Jaccard) das principais formas características entre k-1 e k;
  - cobertura das categorias temáticas (sementes editáveis abaixo).

Ponto de saturação: primeiro k em que as três condições se sustentam.

Uso:
    python run_saturation.py --prepared OUT_DIR [--n-classes 4] [--min-freq 3]
        [--runs 10] [--topn 20] [--thematic-k 3]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from labiialex_pipeline.reinert import run_chd  # noqa: E402

# Sementes das 4 categorias temáticas (formas canônicas; edite conforme o estudo)
CATEGORIAS = {
    "alimentacao": ["comer", "comida", "alimentacao", "alimentação", "fruta", "verdura", "legume"],
    "higiene": ["higiene", "lavar", "lava", "hipoclorito", "sanitário", "cru"],
    "sintomas_gi": ["vomitar", "náusea", "enjoo", "intestino", "preso", "diarreia"],
    "terapias_sne": ["sonda", "suplemento", "gastrostomia"],
}

# Limiares do critério de saturação
THR_NEW_RATIO = 0.05
THR_ARI = 0.80
THR_JACCARD = 0.80


def load(prepared: Path):
    frame = pd.read_csv(prepared / "dtm.csv", sep=";", dtype=str, encoding="utf-8")
    uce_ids = frame.iloc[:, 0].astype(str).tolist()
    forms = list(frame.columns[1:])
    matrix = frame.iloc[:, 1:].to_numpy(dtype=np.int8)
    meta = pd.read_csv(prepared / "uce_meta.csv", sep=";", dtype=str, encoding="utf-8")
    gcol = "grupo" if "grupo" in meta.columns else "uci_id"
    gmap = dict(zip(meta["uce_id"].astype(str), meta[gcol].astype(str)))
    groups = np.array([gmap.get(u, "?") for u in uce_ids])
    return matrix, np.array(forms), np.array(uce_ids), groups


def step_state(matrix, forms, uce_ids, groups, chosen, min_freq, n_classes, topn):
    """Estado lexical com os grupos 'chosen': vocab retido, partição CHD, formas característ."""
    mask = np.isin(groups, chosen)
    sub = matrix[mask]
    col = sub.sum(axis=0)
    keep = col >= min_freq
    vocab = set(forms[keep])
    ids = uce_ids[mask]
    assign, charset = {}, set()
    if keep.sum() >= 2 and mask.sum() >= 10:
        try:
            res = run_chd(sparse.csr_matrix(sub[:, keep]), list(forms[keep]),
                          list(ids), n_classes=n_classes)
            assign = {u: int(c) for u, c in zip(res.uce_ids, res.assignments)}
            terms = [(t.chi2, t.form) for cid in res.characteristic_terms
                     for t in res.characteristic_terms[cid] if t.sign == "+"]
            terms.sort(reverse=True)
            charset = {f for _, f in terms[:topn]}
        except Exception:
            pass
    return vocab, assign, charset


def metrics_for_ordering(matrix, forms, uce_ids, groups, order, min_freq, n_classes, topn):
    rows = []
    prev = None
    for k in range(1, len(order) + 1):
        chosen = order[:k]
        vocab, assign, charset = step_state(matrix, forms, uce_ids, groups,
                                            chosen, min_freq, n_classes, topn)
        rec = {"k": k, "grupos": "+".join(chosen), "vocab": len(vocab),
               "novas": np.nan, "new_ratio": np.nan, "chd_ari": np.nan, "char_jaccard": np.nan}
        # cobertura das categorias
        for cat, seeds in CATEGORIAS.items():
            present = sum(1 for s in seeds if s in vocab)
            rec[f"cat_{cat}"] = round(present / len(seeds), 2)
        if prev is not None:
            new = vocab - prev["vocab_set"]
            rec["novas"] = len(new)
            rec["new_ratio"] = round(len(new) / max(len(vocab), 1), 4)
            common = [u for u in prev["assign"] if u in assign]
            if len(common) >= 5 and prev["assign"] and assign:
                a = [prev["assign"][u] for u in common]
                b = [assign[u] for u in common]
                rec["chd_ari"] = round(float(adjusted_rand_score(a, b)), 4)
            if prev["charset"] and charset:
                inter = len(prev["charset"] & charset)
                union = len(prev["charset"] | charset)
                rec["char_jaccard"] = round(inter / union, 4) if union else np.nan
        rows.append(rec)
        prev = {"vocab_set": vocab, "assign": assign, "charset": charset}
    return rows


def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def _first_sustained(rows, key, thr, direction):
    """Primeiro k (>=2) em que rows[i][key] cruza o limiar e assim se mantém até o fim.

    direction '<' para new_ratio (cai e fica abaixo); '>=' para ARI/Jaccard (atinge
    e fica no/acima). Passos sem valor (NaN) não invalidam a sustentação, mas o
    próprio passo de início precisa ter valor que satisfaça o critério.
    """
    def ok(v):
        if _is_nan(v):
            return True
        return v < thr if direction == "<" else v >= thr

    for i, r in enumerate(rows):
        if r["k"] < 2 or _is_nan(r[key]) or not ok(r[key]):
            continue
        if all(ok(r2[key]) for r2 in rows[i:]):
            return r["k"]
    return None


def saturation_breakdown(rows):
    """Saturação por critério + combinada (k a partir do qual tudo se sustenta)."""
    sat_vocab = _first_sustained(rows, "new_ratio", THR_NEW_RATIO, "<")
    sat_ari = _first_sustained(rows, "chd_ari", THR_ARI, ">=")
    sat_jac = _first_sustained(rows, "char_jaccard", THR_JACCARD, ">=")
    sat_estrut = max(sat_ari, sat_jac) if (sat_ari and sat_jac) else None
    # combinada: primeiro k em que as 3 condições valem e se sustentam juntas
    combinada = None
    for i, r in enumerate(rows):
        if r["k"] < 2:
            continue
        def all_ok(r2):
            return (
                (_is_nan(r2["new_ratio"]) or r2["new_ratio"] < THR_NEW_RATIO)
                and (_is_nan(r2["chd_ari"]) or r2["chd_ari"] >= THR_ARI)
                and (_is_nan(r2["char_jaccard"]) or r2["char_jaccard"] >= THR_JACCARD)
            )
        # exige sinal real (não só NaN) no passo de início
        if _is_nan(r["chd_ari"]) and _is_nan(r["char_jaccard"]):
            continue
        if all(all_ok(r2) for r2 in rows[i:]):
            combinada = r["k"]
            break
    return {"vocabulario": sat_vocab, "chd_estabilidade": sat_ari,
            "formas_caracteristicas": sat_jac, "estrutural": sat_estrut,
            "combinada": combinada}


def saturation_point(rows):
    """Compatibilidade: ponto de saturação combinado (3 critérios juntos)."""
    return saturation_breakdown(rows)["combinada"]


def plot(rows, sat_k, thematic_k, path):
    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, [r["new_ratio"] for r in rows], "o-", label="formas novas (proporção)")
    ax.plot(ks, [r["chd_ari"] for r in rows], "s-", label="estabilidade CHD (Rand)")
    ax.plot(ks, [r["char_jaccard"] for r in rows], "^-", label="formas caract. (Jaccard)")
    ax.axhline(THR_NEW_RATIO, color="gray", ls=":", lw=0.8)
    ax.axhline(THR_ARI, color="gray", ls=":", lw=0.8)
    if thematic_k:
        ax.axvline(thematic_k, color="orange", ls="--", label=f"saturação temática (G{thematic_k})")
    if sat_k:
        ax.axvline(sat_k, color="green", ls="--", label=f"saturação lexical (G{sat_k})")
    ax.set_xlabel("nº de grupos acumulados"); ax.set_ylabel("valor")
    ax.set_title("Saturação lexical incremental (ordem cronológica)")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Saturação lexical incremental")
    p.add_argument("--prepared", required=True)
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--min-freq", type=int, default=3)
    p.add_argument("--runs", type=int, default=10, help="ordens aleatórias")
    p.add_argument("--topn", type=int, default=20)
    p.add_argument("--thematic-k", type=int, default=3, help="grupo da saturação temática declarada")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    prepared = Path(args.prepared)
    out = prepared / "saturation"
    out.mkdir(parents=True, exist_ok=True)
    matrix, forms, uce_ids, groups = load(prepared)
    # ignora UCEs sem rótulo de grupo válido (nan/vazio/?)
    invalid = {"", "nan", "none", "?"}
    uniq = sorted(g for g in set(groups.tolist()) if g and g.lower() not in invalid)
    n_sem_grupo = int(sum(1 for g in groups if (not g) or g.lower() in invalid))
    print(f"{len(uniq)} grupos: {uniq} | {matrix.shape[0]} UCEs"
          f"{f' ({n_sem_grupo} sem grupo, ignoradas)' if n_sem_grupo else ''}"
          f", {matrix.shape[1]} formas")

    # ordem cronológica
    chrono = metrics_for_ordering(matrix, forms, uce_ids, groups, uniq,
                                  args.min_freq, args.n_classes, args.topn)
    bd = saturation_breakdown(chrono)
    sat_chrono = bd["combinada"]
    pd.DataFrame(chrono).to_csv(out / "saturation_chronological.csv", sep=";", index=False, encoding="utf-8")
    plot(chrono, bd["estrutural"] or sat_chrono, args.thematic_k, out / "saturation_curve.png")

    print("\nOrdem cronológica (k | grupos | vocab | new_ratio | CHD_ARI | char_Jaccard):")
    for r in chrono:
        cats = " ".join(f"{c[:4]}={r[f'cat_{c}']:.2f}" for c in CATEGORIAS)
        print(f"  {r['k']} | {r['grupos']:26s} | voc {r['vocab']:4d} | "
              f"new {r['new_ratio']!s:7} | ari {r['chd_ari']!s:7} | jac {r['char_jaccard']!s:7} | {cats}")

    def _fmt(k):
        return f"grupo {k}" if k else "não atingida"
    print("\n>>> Saturação por critério (cronológica):")
    print(f"    vocabulário (formas novas < {THR_NEW_RATIO:.0%}):   {_fmt(bd['vocabulario'])}")
    print(f"    estrutura CHD (Rand >= {THR_ARI:.2f}):           {_fmt(bd['chd_estabilidade'])}")
    print(f"    formas características (Jaccard >= {THR_JACCARD:.2f}): {_fmt(bd['formas_caracteristicas'])}")
    print(f"    ESTRUTURAL (CHD + formas juntas):       {_fmt(bd['estrutural'])}")
    print(f"    COMBINADA (3 critérios juntos):         {_fmt(bd['combinada'])}")
    print(f"    Saturação TEMÁTICA declarada (manual):  grupo {args.thematic_k}")

    # ordens aleatórias: robustez da saturação ESTRUTURAL ao efeito de ordem
    rng = random.Random(args.seed)
    sat_points = []
    for _ in range(args.runs):
        order = uniq[:]
        rng.shuffle(order)
        recs = metrics_for_ordering(matrix, forms, uce_ids, groups, order,
                                    args.min_freq, args.n_classes, args.topn)
        sp = saturation_breakdown(recs)["estrutural"]
        if sp is not None:
            sat_points.append(sp)
    rand_summary = {}
    if sat_points:
        arr = np.array(sat_points)
        rand_summary = {"criterio": "estrutural", "n_orderings": args.runs,
                        "n_saturadas": len(sat_points), "mediana": float(np.median(arr)),
                        "min": int(arr.min()), "max": int(arr.max())}
        print(f"\n>>> Saturação ESTRUTURAL em {args.runs} ordens aleatórias "
              f"({len(sat_points)} saturaram): mediana grupo {np.median(arr):.0f} "
              f"(intervalo {arr.min()}-{arr.max()})")
    else:
        print(f"\n>>> Saturação estrutural em ordens aleatórias: nenhuma das {args.runs} saturou")

    cobertura = {cat: chrono[-1][f"cat_{cat}"] for cat in CATEGORIAS}
    # 1º k em que cada categoria temática fica totalmente coberta e assim permanece
    cobertura_em = {}
    for cat in CATEGORIAS:
        kk = next((r["k"] for i, r in enumerate(chrono)
                   if all(r2[f"cat_{cat}"] >= 0.999 for r2 in chrono[i:])), None)
        cobertura_em[cat] = kk
    summary = {
        "saturacao_lexical_cronologica": sat_chrono,
        "saturacao_por_criterio": bd,
        "saturacao_tematica_declarada": args.thematic_k,
        "saturacao_lexical_aleatoria": rand_summary,
        "limiares": {"new_ratio": THR_NEW_RATIO, "chd_ari": THR_ARI, "char_jaccard": THR_JACCARD},
        "cobertura_categorias_no_corpus_completo": cobertura,
        "categoria_totalmente_coberta_no_grupo": cobertura_em,
    }
    (out / "saturation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados em {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
