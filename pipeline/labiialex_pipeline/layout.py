"""Layout ForceAtlas2 (o algoritmo de disposição de grafos do Gephi).

Implementação compacta para grafos pequenos/médios (sem Barnes-Hut): repulsão
entre todos os pares, atração ao longo das arestas e gravidade ao centro.
"""

from __future__ import annotations

import numpy as np


def force_atlas2(
    n_nodes: int,
    edges: list[tuple[int, int, float]],
    iterations: int = 500,
    gravity: float = 1.0,
    scaling: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """Calcula posições 2D (n_nodes x 2) por ForceAtlas2."""
    rng = np.random.RandomState(seed)
    pos = rng.rand(n_nodes, 2) * 2.0 - 1.0
    degree = np.ones(n_nodes)
    for i, j, _ in edges:
        degree[i] += 1
        degree[j] += 1

    edge_arr = np.array([(i, j, w) for i, j, w in edges], dtype=float) if edges else np.empty((0, 3))
    speed = 0.1
    for _ in range(iterations):
        disp = np.zeros((n_nodes, 2))

        # repulsão (todos os pares): proporcional a (deg_i+1)(deg_j+1)/dist
        delta = pos[:, None, :] - pos[None, :, :]
        dist2 = np.sum(delta ** 2, axis=2) + 1e-9
        np.fill_diagonal(dist2, np.inf)
        coeff = scaling * np.outer(degree, degree)
        rep = (coeff / dist2)[:, :, None] * delta
        disp += rep.sum(axis=1)

        # atração ao longo das arestas (linear, ponderada)
        for i, j, w in edge_arr:
            i, j = int(i), int(j)
            d = pos[i] - pos[j]
            disp[i] -= w * d
            disp[j] += w * d

        # gravidade ao centro
        disp -= gravity * (degree[:, None]) * pos

        # passo limitado
        length = np.sqrt(np.sum(disp ** 2, axis=1)) + 1e-9
        step = np.minimum(length, speed * 10) / length
        pos += disp * step[:, None]

    # normaliza para caber em [-1, 1]
    span = np.abs(pos).max() + 1e-9
    return pos / span
