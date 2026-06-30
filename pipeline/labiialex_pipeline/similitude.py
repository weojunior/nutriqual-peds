"""Análise de similitude (redes de coocorrência de formas) -- método IRaMuTeQ.

Correções em relação ao ``simi.R`` original:

* célula ``d`` da tabela 2x2 = ``N - a - b - c`` com N = nº de UCEs (``nrow``),
  e não ``ncol`` (nº de formas), que produzia ``d`` errado/negativo;
* a matriz de coocorrência é construída a partir da matriz BINÁRIA (presença por
  UCE), e não de contagens, evitando inflar as arestas;
* coeficientes (jaccard, dice, cosseno, phi) bem definidos sobre a 2x2 correta.

A árvore máxima (``arbre maximum``) é o grafo de similitude canônico: liga todas
as formas pelos vínculos mais fortes sem ciclos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

#: Índices de associação suportados.
INDICES = ("cooccurrence", "jaccard", "dice", "cosine", "phi")


@dataclass
class SimilitudeGraph:
    forms: list[str]
    frequency: np.ndarray            # nº de UCEs em que cada forma aparece
    cooccurrence: np.ndarray         # matriz forma x forma (coocorrência em UCEs)
    similarity: np.ndarray           # índice de associação escolhido
    mst_edges: list[tuple[int, int, float]]   # (i, j, peso) da árvore máxima
    communities: np.ndarray          # rótulo de comunidade por forma
    index: str


def cooccurrence_matrix(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coocorrência forma x forma a partir da matriz binária UCE x forma."""
    binary = (np.asarray(binary) > 0).astype(float)   # garante binarização
    cooc = binary.T @ binary                           # a[i,j] = UCEs com i e j
    frequency = np.diag(cooc).copy()
    return cooc, frequency


def association_matrix(
    cooc: np.ndarray, frequency: np.ndarray, n_uce: int, index: str = "cooccurrence"
) -> np.ndarray:
    """Índice de associação entre formas a partir da 2x2 (a,b,c,d) CORRETA."""
    if index not in INDICES:
        raise ValueError(f"Índice desconhecido: {index}")
    a = cooc.astype(float)
    fi = frequency[:, None]
    fj = frequency[None, :]
    b = fi - a
    c = fj - a
    d = n_uce - a - b - c            # CORREÇÃO: N de UCEs (nrow), não nº de formas
    with np.errstate(divide="ignore", invalid="ignore"):
        if index == "cooccurrence":
            sim = a.copy()
        elif index == "jaccard":
            sim = a / (a + b + c)
        elif index == "dice":
            sim = 2 * a / (2 * a + b + c)
        elif index == "cosine":
            sim = a / np.sqrt((a + b) * (a + c))
        else:  # phi (precisa de d correto)
            num = a * d - b * c
            den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
            sim = num / den
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(sim, 0.0)
    return sim


def maximum_spanning_tree(similarity: np.ndarray) -> list[tuple[int, int, float]]:
    """Árvore máxima: MST sobre o negativo da similaridade (apenas pesos > 0)."""
    sim = np.where(similarity > 0, similarity, 0.0)
    mst = minimum_spanning_tree(-sim)        # minimiza -sim = maximiza sim
    coo = mst.tocoo()
    edges = [(int(i), int(j), float(-w)) for i, j, w in zip(coo.row, coo.col, coo.data)]
    return edges


def detect_communities(n_nodes: int, edges: list[tuple[int, int, float]]) -> np.ndarray:
    """Comunidades (Louvain via igraph) sobre o grafo da árvore máxima."""
    try:
        import igraph
    except ImportError:
        return np.zeros(n_nodes, dtype=int)
    g = igraph.Graph(n=n_nodes, edges=[(i, j) for i, j, _ in edges])
    g.es["weight"] = [w for _, _, w in edges]
    if g.ecount() == 0:
        return np.zeros(n_nodes, dtype=int)
    membership = g.community_multilevel(weights="weight").membership
    return np.asarray(membership, dtype=int)


def build_similitude(
    binary: np.ndarray,
    forms: list[str],
    n_uce: int,
    index: str = "cooccurrence",
) -> SimilitudeGraph:
    """Pipeline completo: coocorrência -> associação -> árvore máxima -> comunidades."""
    cooc, frequency = cooccurrence_matrix(binary)
    sim = association_matrix(cooc, frequency, n_uce, index=index)
    edges = maximum_spanning_tree(sim)
    communities = detect_communities(len(forms), edges)
    return SimilitudeGraph(
        forms=forms, frequency=frequency, cooccurrence=cooc, similarity=sim,
        mst_edges=edges, communities=communities, index=index,
    )


def phi_buggy_vs_correct(
    cooc: np.ndarray, frequency: np.ndarray, n_uce: int, n_forms: int
) -> dict:
    """Demonstra o bug do simi.R: d = nº de formas (errado) vs N de UCEs (correto)."""
    a = cooc.astype(float)
    b = frequency[:, None] - a
    c = frequency[None, :] - a

    def phi(d):
        with np.errstate(divide="ignore", invalid="ignore"):
            num = a * d - b * c
            den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
            return np.nan_to_num(num / den, nan=0.0, posinf=0.0, neginf=0.0)

    correct = phi(n_uce - a - b - c)        # d com nrow (UCEs) -- correto
    buggy = phi(n_forms - a - b - c)        # d com ncol (formas) -- bug original
    off = np.triu_indices_from(a, k=1)
    diff = np.abs(correct[off] - buggy[off])
    n_neg_buggy = int(np.sum((n_forms - a - b - c)[off] < 0))
    return {
        "max_abs_diff_phi": round(float(diff.max()) if diff.size else 0.0, 4),
        "mean_abs_diff_phi": round(float(diff.mean()) if diff.size else 0.0, 4),
        "n_pairs_d_negative_no_bug": n_neg_buggy,
    }
