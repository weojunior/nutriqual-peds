#!/usr/bin/env python3
"""Parte 3: AFC (classes x formas) + especificidades por variável, Python x R.

Pré-requisito: ter rodado a Parte 2 (run_chd.py) -- usa chd/chd_classes.csv.

Uso:
    python run_afc_spec.py --prepared OUT_DIR [--n-classes 3] [--variables tema,grupo]

Saídas em OUT_DIR/afc_spec/:
    afc_coords_forms.csv / afc_coords_classes.csv   coordenadas fatoriais
    afc_inertia.csv                                 autovalores e % de inércia (Py x R)
    afc_plane.png                                   plano fatorial (classes + formas)
    specificities_<var>.csv                         especificidades por modalidade (Py)
    afc_spec_comparison.json                        cruzamento Python x R
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labiialex_pipeline.afc import class_form_table, correspondence_analysis  # noqa: E402
from labiialex_pipeline.specificities import (  # noqa: E402
    build_form_part_table,
    specificities,
)

R_DIR = Path(__file__).resolve().parent / "r"
META_NON_VARS = {"uce_id", "uci_id", "n_tokens", "n_active", "classe_py", "classe_r"}


def load_dtm(path: Path):
    frame = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8")
    uce_ids = frame.iloc[:, 0].astype(str).tolist()
    forms = list(frame.columns[1:])
    values = frame.iloc[:, 1:].to_numpy(dtype=np.int8)
    return values, forms, uce_ids


def write_table_csv(table: np.ndarray, rows: list[str], cols: list[str], path: Path):
    frame = pd.DataFrame(table.astype(int), index=rows, columns=cols)
    frame.to_csv(path, sep=";", encoding="utf-8")


def run_r(script: str, *script_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["Rscript", str(R_DIR / script), *map(str, script_args)],
        capture_output=True, text=True, encoding="utf-8",
    )


def do_afc(dtm, forms, assignments, n_classes, out_dir) -> dict:
    table, row_labels = class_form_table(dtm.astype(float), assignments, forms, n_classes)
    keep = table.sum(axis=1) > 0
    table, row_labels = table[keep], [f for f, k in zip(forms, keep) if k]
    col_labels = [f"classe{c}" for c in range(1, n_classes + 1)]
    res = correspondence_analysis(table, row_labels, col_labels)

    def _points_table(labels, coords, cor, ctr, mass):
        """Coordenadas + COR (qualidade, cos^2) + CTR (contribuição) + massa."""
        k = min(3, res.n_axes)
        data = {}
        for i in range(k):
            data[f"eixo{i+1}"] = np.round(coords[:, i], 4)
        for i in range(k):
            data[f"cor{i+1}"] = np.round(cor[:, i], 4)        # qualidade no eixo i
        for i in range(k):
            data[f"ctr{i+1}"] = np.round(ctr[:, i], 4)        # contribuição ao eixo i
        data["massa"] = np.round(mass, 5)
        data["qualidade_2d"] = np.round(cor[:, :2].sum(axis=1), 4)  # COR nos eixos 1+2
        return pd.DataFrame(data, index=labels)

    _points_table(row_labels, res.row_coords, res.row_cor, res.row_contrib,
                  res.row_mass).to_csv(out_dir / "afc_coords_forms.csv",
                                       sep=";", encoding="utf-8")
    _points_table(col_labels, res.col_coords, res.col_cor, res.col_contrib,
                  res.col_mass).to_csv(out_dir / "afc_coords_classes.csv",
                                       sep=";", encoding="utf-8")

    # cruzamento com o pacote 'ca' do R
    table_path = out_dir / "_afc_table.csv"
    write_table_csv(table, row_labels, col_labels, table_path)
    eig_r_path = out_dir / "afc_inertia_r.csv"
    proc = run_r("afc_reference.R", table_path, eig_r_path)
    print(proc.stdout.strip() or proc.stderr.strip()[:300])

    inertia = pd.DataFrame({
        "eixo": np.arange(1, res.n_axes + 1),
        "autovalor_py": np.round(res.eigenvalues, 8),
        "inercia_pct_py": np.round(res.inertia_pct, 4),
    })
    afc_cmp = {"total_inertia_py": round(res.total_inertia, 6),
               "inertia_pct_py": [round(float(x), 3) for x in res.inertia_pct]}
    if eig_r_path.exists():
        r_eig = pd.read_csv(eig_r_path, sep=";")
        inertia = inertia.merge(
            r_eig.rename(columns={"autovalor": "autovalor_r", "inercia_pct": "inercia_pct_r"}),
            on="eixo", how="left",
        )
        afc_cmp["inertia_pct_r"] = [round(float(x), 3) for x in r_eig["inercia_pct"]]
        afc_cmp["max_abs_diff_inertia_pct"] = round(
            float(np.nanmax(np.abs(inertia["inercia_pct_py"] - inertia["inercia_pct_r"]))), 4
        )
    inertia.to_csv(out_dir / "afc_inertia.csv", sep=";", index=False, encoding="utf-8")

    _plot_plane(res, assignments, n_classes, out_dir / "afc_plane.png")
    return afc_cmp


def _plot_plane(res, assignments, n_classes, path):
    if res.n_axes < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    # formas: top 10 por contribuição em cada eixo
    contrib = res.row_contrib[:, 0] + res.row_contrib[:, 1]
    top = np.argsort(contrib)[::-1][:30]
    ax.scatter(res.row_coords[top, 0], res.row_coords[top, 1], s=8, c="gray", alpha=0.6)
    for i in top:
        ax.annotate(res.row_labels[i], (res.row_coords[i, 0], res.row_coords[i, 1]),
                    fontsize=7, color="dimgray")
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for c in range(n_classes):
        ax.scatter(res.col_coords[c, 0], res.col_coords[c, 1], s=180,
                   marker="*", color=colors[c], edgecolor="black", zorder=5,
                   label=res.col_labels[c])
    ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel(f"Eixo 1 ({res.inertia_pct[0]:.1f}%)")
    ax.set_ylabel(f"Eixo 2 ({res.inertia_pct[1]:.1f}%)")
    ax.set_title("AFC das classes (formas + classes)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def do_specificities(dtm, forms, meta, variables, out_dir) -> dict:
    cmp: dict = {}
    for var in variables:
        labels = meta[var].astype(str).tolist()
        if len(set(labels)) < 2:
            continue
        table, parts = build_form_part_table(dtm.astype(float), labels, forms)
        sp = specificities(table, forms, parts, min_score=2.0)
        pd.DataFrame([s.__dict__ for s in sp]).to_csv(
            out_dir / f"specificities_{var}.csv", sep=";", index=False, encoding="utf-8"
        )
        # cruzamento com textometry
        table_path = out_dir / f"_spec_table_{var}.csv"
        write_table_csv(table, forms, parts, table_path)
        r_out = out_dir / f"specificities_{var}_r.csv"
        proc = run_r("specif_reference.R", table_path, r_out)
        print(proc.stdout.strip() or proc.stderr.strip()[:300])
        if r_out.exists():
            py_long = {(s.form, s.part): (s.score if s.sign == "+" else -s.score) for s in sp}
            r_df = pd.read_csv(r_out, sep=";")
            pairs = []
            for _, row in r_df.iterrows():
                key = (str(row["forma"]), str(row["parte"]))
                if key in py_long:
                    pairs.append((py_long[key], float(row["score"])))
            if len(pairs) >= 3:
                arr = np.array(pairs)
                corr = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
                sign_agree = float(np.mean(np.sign(arr[:, 0]) == np.sign(arr[:, 1])))
                cmp[var] = {"n_compared": len(pairs),
                            "pearson_py_vs_r": round(corr, 4),
                            "sign_agreement": round(sign_agree, 4)}
    return cmp


def main() -> int:
    parser = argparse.ArgumentParser(description="AFC + especificidades (Python x R)")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--n-classes", type=int, default=0, help="0 = inferir do CHD")
    parser.add_argument("--variables", default="", help="lista separada por vírgula; vazio = todas")
    args = parser.parse_args()

    prepared = Path(args.prepared)
    out_dir = prepared / "afc_spec"
    out_dir.mkdir(parents=True, exist_ok=True)

    dtm_vals, forms, uce_ids = load_dtm(prepared / "dtm.csv")
    chd_path = prepared / "chd" / "chd_classes.csv"
    if not chd_path.exists():
        print("Erro: rode a Parte 2 (run_chd.py) antes."); return 2
    meta = pd.read_csv(chd_path, sep=";", dtype=str, encoding="utf-8")
    meta = meta.set_index("uce_id").reindex(uce_ids).reset_index()
    assignments = meta["classe_py"].astype(int).to_numpy()
    n_classes = args.n_classes or int(assignments.max())

    print(f"AFC sobre {dtm_vals.shape[1]} formas x {n_classes} classes ...")
    afc_cmp = do_afc(dtm_vals, forms, assignments, n_classes, out_dir)
    if "inertia_pct_r" in afc_cmp:
        print(f"  Inércia %% Python: {afc_cmp['inertia_pct_py']}")
        print(f"  Inércia %% R(ca): {afc_cmp['inertia_pct_r']}  "
              f"(maior diferença {afc_cmp['max_abs_diff_inertia_pct']} pp)")

    var_cols = [c for c in meta.columns if c not in META_NON_VARS]
    if args.variables:
        var_cols = [v.strip() for v in args.variables.split(",") if v.strip()]
    print(f"Especificidades por variável: {var_cols}")
    spec_cmp = do_specificities(dtm_vals, forms, meta, var_cols, out_dir)
    for var, c in spec_cmp.items():
        print(f"  {var}: correlação Py x R = {c['pearson_py_vs_r']}, "
              f"concordância de sinal = {c['sign_agreement']} (n={c['n_compared']})")

    (out_dir / "afc_spec_comparison.json").write_text(
        json.dumps({"afc": afc_cmp, "specificities": spec_cmp}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultados em {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
