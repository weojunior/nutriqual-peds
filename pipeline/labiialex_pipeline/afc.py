"""Análise Fatorial de Correspondência (AFC/CA) sobre a tabela classes x formas.

CA clássica de Benzécri por SVD dos resíduos padronizados. Projeta as classes da
CHD e as formas ativas nos eixos fatoriais.

Correção em relação ao código original: a inércia explicada por eixo é calculada
sobre a inércia TOTAL (soma de todos os autovalores), não sobre os eixos retidos
(``reinert/engine.py`` dividia pela soma truncada e inflava os percentuais).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AfcResult:
    eigenvalues: np.ndarray          # autovalores (lambda_k) de todos os eixos
    inertia_pct: np.ndarray          # % de inércia por eixo (sobre a inércia total)
    total_inertia: float
    row_coords: np.ndarray           # coords principais das linhas (formas)
    col_coords: np.ndarray           # coords principais das colunas (classes)
    row_contrib: np.ndarray          # CTR: contribuição de cada linha a cada eixo
    row_labels: list[str]
    col_labels: list[str]
    row_cor: np.ndarray = None       # COR: qualidade (cos^2) de cada linha em cada eixo
    col_contrib: np.ndarray = None   # CTR das colunas
    col_cor: np.ndarray = None       # COR das colunas
    row_mass: np.ndarray = None      # massa de cada linha
    col_mass: np.ndarray = None      # massa de cada coluna

    @property
    def n_axes(self) -> int:
        return self.eigenvalues.size


def correspondence_analysis(
    table: np.ndarray, row_labels: list[str], col_labels: list[str]
) -> AfcResult:
    """CA de uma tabela de contingência (linhas x colunas), ambos > 0."""
    table = np.asarray(table, dtype=float)
    total = table.sum()
    if total <= 0:
        raise ValueError("Tabela vazia para a AFC.")
    p = table / total
    r = p.sum(axis=1)               # massas das linhas
    c = p.sum(axis=0)               # massas das colunas
    r_safe = np.where(r == 0, 1e-12, r)
    c_safe = np.where(c == 0, 1e-12, c)

    # resíduos padronizados: S = D_r^{-1/2} (P - r c^T) D_c^{-1/2}
    residual = (p - np.outer(r, c)) / np.sqrt(np.outer(r_safe, c_safe))
    u, sv, vt = np.linalg.svd(residual, full_matrices=False)

    # autovalores e inércia total (soma de TODOS os autovalores não triviais)
    eig_all = sv ** 2
    nontrivial = eig_all > 1e-12
    eig = eig_all[nontrivial]
    u = u[:, nontrivial]
    sv = sv[nontrivial]
    vt = vt[nontrivial, :]
    total_inertia = float(eig.sum())
    inertia_pct = 100.0 * eig / total_inertia if total_inertia > 0 else eig * 0.0

    # coordenadas principais
    row_coords = (u * sv) / np.sqrt(r_safe)[:, None]
    col_coords = (vt.T * sv) / np.sqrt(c_safe)[:, None]

    # CTR: contribuição de cada ponto ao eixo = massa * coord^2 / autovalor (soma 1 por eixo)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(eig > 0, eig, np.nan)
        row_contrib = np.nan_to_num((r[:, None] * row_coords ** 2) / denom)
        col_contrib = np.nan_to_num((c[:, None] * col_coords ** 2) / denom)

    # COR: qualidade da representação (cos^2) = coord^2 / soma dos coord^2 do ponto
    def _cos2(coords: np.ndarray) -> np.ndarray:
        tot = (coords ** 2).sum(axis=1, keepdims=True)
        return np.nan_to_num(coords ** 2 / np.where(tot > 0, tot, np.nan))

    return AfcResult(
        eigenvalues=eig,
        inertia_pct=inertia_pct,
        total_inertia=total_inertia,
        row_coords=row_coords,
        col_coords=col_coords,
        row_contrib=row_contrib,
        row_labels=row_labels,
        col_labels=col_labels,
        row_cor=_cos2(row_coords),
        col_contrib=col_contrib,
        col_cor=_cos2(col_coords),
        row_mass=r,
        col_mass=c,
    )


def class_form_table(
    dtm: np.ndarray, assignments: np.ndarray, forms: list[str], n_classes: int
) -> tuple[np.ndarray, list[str]]:
    """Tabela formas x classes: nº de UCEs de cada classe que contêm a forma."""
    table = np.zeros((len(forms), n_classes), dtype=float)
    for class_id in range(1, n_classes + 1):
        mask = assignments == class_id
        table[:, class_id - 1] = (dtm[mask] > 0).sum(axis=0)
    return table, forms
