"""Especificidades de Lafon (modelo hipergeométrico, método do textometry/Lexico).

Para cada forma f e cada parte (modalidade de uma variável), mede se a forma é
sobre ou sub-representada na parte em relação ao conjunto, pela distribuição
hipergeométrica:

* N  = soma da tabela (total de presenças forma x parte)
* Kf = total da linha (forma f no conjunto)
* np = total da coluna (parte p)
* k  = célula observada (forma f na parte p)

Especificidade (escore assinado) = -log10 P(X >= k)  se k >= esperado  (sinal +)
                                   = +log10 P(X <= k)  caso contrário    (sinal -)

A tabela forma x parte é a MESMA passada ao textometry::specificities no R,
permitindo cruzamento exato.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import hypergeom

#: Limite numérico para evitar log10(0).
_MIN_P = 1e-300


@dataclass
class Specificity:
    form: str
    part: str
    score: float       # -log10(p) assinado
    sign: str          # "+" (sobre) ou "-" (sub)
    observed: int
    expected: float


def build_form_part_table(
    dtm: np.ndarray, labels: list[str], forms: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Tabela forma x parte: nº de UCEs de cada parte que contêm a forma."""
    parts = sorted(set(labels))
    part_index = {p: j for j, p in enumerate(parts)}
    table = np.zeros((len(forms), len(parts)), dtype=float)
    present = (dtm > 0).astype(float)
    for i, lab in enumerate(labels):
        table[:, part_index[lab]] += present[i]
    return table, parts


def specificities(
    table: np.ndarray, forms: list[str], parts: list[str], min_score: float = 2.0
) -> list[Specificity]:
    """Calcula especificidades de Lafon para cada célula relevante da tabela."""
    table = np.asarray(table, dtype=float)
    grand_total = float(table.sum())
    if grand_total <= 0:
        return []
    row_tot = table.sum(axis=1)   # Kf por forma
    col_tot = table.sum(axis=0)   # np por parte
    n_pop = int(round(grand_total))

    out: list[Specificity] = []
    for j, part in enumerate(parts):
        draws = int(round(col_tot[j]))
        if draws <= 0:
            continue
        for i, form in enumerate(forms):
            kf = int(round(row_tot[i]))
            k = int(round(table[i, j]))
            if kf <= 0:
                continue
            expected = kf * draws / grand_total
            if k >= expected:
                # P(X >= k) = sf(k-1)
                p = float(hypergeom.sf(k - 1, n_pop, kf, draws))
                score = -np.log10(max(p, _MIN_P))
                sign = "+"
            else:
                p = float(hypergeom.cdf(k, n_pop, kf, draws))
                score = -np.log10(max(p, _MIN_P))
                sign = "-"
            if score >= min_score:
                out.append(
                    Specificity(
                        form=form, part=part, score=round(float(score), 3),
                        sign=sign, observed=k, expected=round(float(expected), 3),
                    )
                )
    out.sort(key=lambda s: (s.part, -s.score))
    return out
