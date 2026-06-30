"""Dendrograma da CHD (árvore de divisões descendentes).

A CHD de Reinert é divisiva: a cada passo a maior classe é bipartida. Este módulo
desenha a árvore resultante como um filograma (estilo IRaMuTeQ): a raiz no topo, as
classes terminais embaixo, a altura de cada nó dada pela ordem da divisão e o
qui-quadrado do corte anotado. As folhas trazem o rótulo da classe, o tamanho e as
principais formas características.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _leaf_label(result, leaf_node: dict, top_terms: int) -> str:
    cls = leaf_node.get("label")
    size = leaf_node.get("size", 0)
    terms = result.characteristic_terms.get(cls, [])
    words = [t.form for t in terms if t.sign == "+"][:top_terms]
    head = f"Classe {cls}\n(n={size})"
    return head + ("\n" + "\n".join(words) if words else "")


def render_dendrogram(result, path: str | Path, top_terms: int = 5,
                      title: str = "Dendrograma da CHD (Reinert)") -> Path:
    """Desenha o dendrograma da árvore descendente em ``result.tree``."""
    tree = result.tree
    if not tree or tree[0].get("children") is None:
        raise ValueError("Árvore da CHD vazia ou sem divisões para desenhar.")

    internal = [i for i, n in enumerate(tree) if n["children"] is not None]
    n_int = len(internal)

    def height(i: int) -> float:
        node = tree[i]
        if node["children"] is None:
            return 0.0
        return float(n_int - node["order"] + 1)

    # posição horizontal das folhas por percurso em ordem (sem cruzamentos)
    xpos: dict[int, float] = {}
    counter = [0]

    def assign_x(i: int) -> float:
        node = tree[i]
        if node["children"] is None:
            xpos[i] = float(counter[0])
            counter[0] += 1
            return xpos[i]
        a, b = (assign_x(c) for c in node["children"])
        xpos[i] = (a + b) / 2.0
        return xpos[i]

    assign_x(0)
    n_leaves = counter[0]

    fig, ax = plt.subplots(figsize=(max(7, 1.7 * n_leaves), 6))

    def draw(i: int) -> None:
        node = tree[i]
        if node["children"] is None:
            return
        h = height(i)
        c1, c2 = node["children"]
        for c in (c1, c2):
            draw(c)
            ax.plot([xpos[c], xpos[c]], [height(c), h], color="#333333", lw=1.4)
        ax.plot([xpos[c1], xpos[c2]], [h, h], color="#333333", lw=1.4)
        ax.annotate(f"χ²={node['chi2']:.0f}", (xpos[i], h),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontsize=8, color="#1f4e79")

    draw(0)

    for i in (j for j, n in enumerate(tree) if n["children"] is None):
        ax.text(xpos[i], -0.4, _leaf_label(result, tree[i], top_terms),
                ha="center", va="top", fontsize=8.5)

    ax.set_xlim(-0.7, n_leaves - 0.3)
    ax.set_ylim(-3.2, n_int + 0.8)
    ax.set_title(title)
    ax.set_yticks(range(1, n_int + 1))
    ax.set_ylabel("ordem da divisão (raiz no topo)")
    ax.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
