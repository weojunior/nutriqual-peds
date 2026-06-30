#!/usr/bin/env python3
"""Parte 4: análise de similitude (rede de coocorrência), Python x R (igraph).

Uso:
    python run_simi.py --prepared OUT_DIR [--index cooccurrence] [--top 60]

Saídas em OUT_DIR/simi/:
    simi_nodes.csv     forma, frequência, comunidade
    simi_edges.csv     forma_i, forma_j, coocorrência, phi
    simi_graph.png     árvore máxima de similitude (cor = comunidade)
    simi_comparison.json   cruzamento Python x R + demonstração do bug do simi.R
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labiialex_pipeline.layout import force_atlas2  # noqa: E402
from labiialex_pipeline.similitude import (  # noqa: E402
    build_similitude,
    phi_buggy_vs_correct,
)

R_DIR = Path(__file__).resolve().parent / "r"


def load_dtm(path: Path):
    frame = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8")
    forms = list(frame.columns[1:])
    values = frame.iloc[:, 1:].to_numpy(dtype=np.int8)
    return values, forms


def build_nx_graph(graph):
    """Grafo networkx da árvore máxima, com atributos (para plot e GEXF)."""
    g = nx.Graph()
    for i, form in enumerate(graph.forms):
        g.add_node(i, label=form, forma=form,
                   frequencia=int(graph.frequency[i]),
                   comunidade=int(graph.communities[i]))
    for i, j, w in graph.mst_edges:
        g.add_edge(i, j, weight=float(w))
    return g


def compute_positions(graph, g, layout: str):
    if layout == "fa2":
        coords = force_atlas2(len(graph.forms), graph.mst_edges)
        return {i: (float(coords[i, 0]), float(coords[i, 1])) for i in g.nodes()}
    return nx.spring_layout(g, weight="weight", seed=42, k=0.6)


def plot_graph(graph, g, pos, freq, communities, out_path):
    fig, ax = plt.subplots(figsize=(11, 9))
    sizes = 120 + 700 * (freq / freq.max())
    palette = plt.cm.tab10(np.linspace(0, 1, max(communities) + 1))
    nx.draw_networkx_edges(g, pos, ax=ax, width=[1 + 2 * g[u][v]["weight"] / max(1, freq.max()) for u, v in g.edges()], edge_color="lightgray")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=sizes,
                           node_color=[palette[communities[n]] for n in g.nodes()])
    nx.draw_networkx_labels(g, pos, ax=ax,
                            labels={i: graph.forms[i] for i in g.nodes()}, font_size=8)
    ax.set_title("Análise de similitude (árvore máxima; cor = comunidade)")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Similitude (Python x R igraph)")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--index", default="cooccurrence",
                        choices=["cooccurrence", "jaccard", "dice", "cosine", "phi"])
    parser.add_argument("--top", type=int, default=60, help="nº de formas mais frequentes")
    parser.add_argument("--layout", choices=["fa2", "spring"], default="fa2",
                        help="fa2 = ForceAtlas2 (Gephi); spring = Fruchterman-Reingold")
    args = parser.parse_args()

    prepared = Path(args.prepared)
    out_dir = prepared / "simi"
    out_dir.mkdir(parents=True, exist_ok=True)

    values, forms = load_dtm(prepared / "dtm.csv")
    n_uce = values.shape[0]
    freq_all = (values > 0).sum(axis=0)
    keep = np.argsort(freq_all)[::-1][: min(args.top, len(forms))]
    keep = np.sort(keep)
    sub = values[:, keep]
    sub_forms = [forms[i] for i in keep]
    print(f"Similitude sobre {len(sub_forms)} formas (top por frequência), "
          f"{n_uce} UCEs, índice '{args.index}'.")

    graph = build_similitude(sub, sub_forms, n_uce, index=args.index)

    # demonstração do bug do simi.R (d com ncol vs nrow)
    bug = phi_buggy_vs_correct(graph.cooccurrence, graph.frequency, n_uce, len(sub_forms))
    print(f"Bug simi.R (d=ncol vs nrow): maior diferença em phi = {bug['max_abs_diff_phi']}; "
          f"pares com d negativo no código bugado = {bug['n_pairs_d_negative_no_bug']}")

    # nós e arestas
    pd.DataFrame({
        "forma": graph.forms,
        "frequencia": graph.frequency.astype(int),
        "comunidade": graph.communities,
    }).to_csv(out_dir / "simi_nodes.csv", sep=";", index=False, encoding="utf-8")
    edge_rows = [
        {"forma_i": graph.forms[i], "forma_j": graph.forms[j],
         "peso": round(w, 4),
         "coocorrencia": int(graph.cooccurrence[i, j])}
        for i, j, w in graph.mst_edges
    ]
    pd.DataFrame(edge_rows).to_csv(out_dir / "simi_edges.csv", sep=";", index=False, encoding="utf-8")

    g = build_nx_graph(graph)
    pos = compute_positions(graph, g, args.layout)
    plot_graph(graph, g, pos, graph.frequency, graph.communities, out_dir / "simi_graph.png")
    # GEXF para abrir no Gephi (atributos: forma, frequência, comunidade; peso nas arestas)
    nx.write_gexf(g, out_dir / "simi_graph.gexf")
    print(f"GEXF para o Gephi: {out_dir / 'simi_graph.gexf'} (layout '{args.layout}')")

    # cruzamento com R igraph (mesma matriz de similaridade -> árvore máxima)
    sim_path = out_dir / "_sim_matrix.csv"
    pd.DataFrame(graph.similarity, index=graph.forms, columns=graph.forms).to_csv(
        sim_path, sep=";", encoding="utf-8")
    r_edges = out_dir / "simi_edges_r.csv"
    proc = subprocess.run(["Rscript", str(R_DIR / "simi_reference.R"), str(sim_path), str(r_edges)],
                          capture_output=True, text=True, encoding="utf-8")
    print(proc.stdout.strip() or proc.stderr.strip()[:300])

    comparison = {"n_forms": len(sub_forms), "n_uce": int(n_uce),
                  "index": args.index, "bug_demo": bug,
                  "n_communities": int(graph.communities.max() + 1)}
    py_w = sum(w for _, _, w in graph.mst_edges)
    comparison["python_mst_total_weight"] = round(float(py_w), 4)
    if r_edges.exists():
        r_df = pd.read_csv(r_edges, sep=";")
        comparison["r_mst_total_weight"] = round(float(r_df["peso"].sum()), 4)
        py_set = {frozenset((graph.forms[i], graph.forms[j])) for i, j, _ in graph.mst_edges}
        r_set = {frozenset((str(a), str(b))) for a, b in zip(r_df["forma_i"], r_df["forma_j"])}
        inter = len(py_set & r_set)
        union = len(py_set | r_set)
        comparison["mst_edge_jaccard_py_vs_r"] = round(inter / union, 4) if union else 0.0
        print(f"\n>>> Árvore máxima: peso total Python={py_w:.2f} vs R={r_df['peso'].sum():.2f}; "
              f"arestas em comum (Jaccard) = {comparison['mst_edge_jaccard_py_vs_r']}")

    (out_dir / "simi_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nComunidades encontradas: {comparison['n_communities']}")
    nodes = pd.read_csv(out_dir / "simi_nodes.csv", sep=";")
    for com in sorted(nodes["comunidade"].unique()):
        top = nodes[nodes["comunidade"] == com].nlargest(8, "frequencia")["forma"].tolist()
        print(f"  Comunidade {com}: {', '.join(top)}")
    print(f"\nResultados em {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
