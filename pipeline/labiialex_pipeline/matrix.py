"""Construção da matriz UCE x formas ativas e exportação para os motores R/Python.

A CHD de Reinert opera sobre uma matriz binária (presença/ausência) de formas
ativas por UCE, mantendo apenas formas com frequência global >= ``min_freq``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from .corpus import Uce
from .errors import MatrixError

#: Frequência mínima global de uma forma para entrar na matriz (tradição Reinert).
DEFAULT_MIN_FREQ: int = 3

#: Número mínimo de UCEs para uma análise minimamente estável.
MIN_UCES: int = 4


@dataclass
class DocTermMatrix:
    """Matriz esparsa binária UCE x forma e seus rótulos."""

    matrix: sparse.csr_matrix  # shape (n_uces, n_forms), valores 0/1
    forms: list[str]
    uce_ids: list[str]
    form_freq: dict[str, int]  # frequência global (ocorrências)
    form_nuce: dict[str, int]  # número de UCEs em que a forma aparece

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape


def build_dtm(uces: list[Uce], min_freq: int = DEFAULT_MIN_FREQ) -> DocTermMatrix:
    """Constrói a matriz binária UCE x forma ativa filtrada por frequência."""
    if len(uces) < MIN_UCES:
        raise MatrixError(
            f"Apenas {len(uces)} UCEs; mínimo {MIN_UCES}. Corpus pequeno demais."
        )

    global_freq: Counter[str] = Counter()
    per_uce_sets: list[set[str]] = []
    for uce in uces:
        global_freq.update(uce.active_lemmas)
        per_uce_sets.append(set(uce.active_lemmas))

    forms = sorted(f for f, c in global_freq.items() if c >= min_freq)
    if not forms:
        raise MatrixError(
            f"Nenhuma forma ativa com frequência >= {min_freq}. "
            "Reduza min_freq ou amplie o corpus."
        )
    form_index = {f: j for j, f in enumerate(forms)}

    rows: list[int] = []
    cols: list[int] = []
    form_nuce: Counter[str] = Counter()
    for i, present in enumerate(per_uce_sets):
        for form in present:
            j = form_index.get(form)
            if j is not None:
                rows.append(i)
                cols.append(j)
                form_nuce[form] += 1

    data = np.ones(len(rows), dtype=np.int8)
    matrix = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(uces), len(forms)), dtype=np.int8
    )

    # Remove UCEs vazias (sem nenhuma forma ativa retida): degenerariam a CHD.
    nonempty = np.asarray(matrix.sum(axis=1)).ravel() > 0
    if not nonempty.all():
        matrix = matrix[nonempty]
    kept_ids = [u.uce_id for u, keep in zip(uces, nonempty) if keep]

    return DocTermMatrix(
        matrix=matrix.tocsr(),
        forms=forms,
        uce_ids=kept_ids,
        form_freq={f: global_freq[f] for f in forms},
        form_nuce=dict(form_nuce),
    )


def build_count_matrix(
    uces: list[Uce], vocabulary: list[str], level: str = "uci"
) -> tuple[np.ndarray, list[str]]:
    """Matriz de CONTAGENS documento x forma (para LDA), sobre um vocabulário fixo.

    ``level='uci'`` agrega por documento (entrevista); ``level='uce'`` por segmento.
    """
    from collections import Counter

    vocab_index = {f: j for j, f in enumerate(vocabulary)}
    buckets: dict[str, Counter[str]] = {}
    order: list[str] = []
    for uce in uces:
        doc_id = uce.uci_id if level == "uci" else uce.uce_id
        if doc_id not in buckets:
            buckets[doc_id] = Counter()
            order.append(doc_id)
        buckets[doc_id].update(uce.active_lemmas)
    matrix = np.zeros((len(order), len(vocabulary)), dtype=int)
    for i, doc_id in enumerate(order):
        for form, count in buckets[doc_id].items():
            j = vocab_index.get(form)
            if j is not None:
                matrix[i, j] = count
    return matrix, order


def export_for_r(dtm: DocTermMatrix, out_dir: str | Path) -> dict[str, Path]:
    """Exporta a matriz e os rótulos em CSV para os scripts R canônicos."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dense = dtm.matrix.toarray()

    dtm_path = out_dir / "dtm.csv"
    with dtm_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("uce_id;" + ";".join(dtm.forms) + "\n")
        for uce_id, row in zip(dtm.uce_ids, dense):
            handle.write(uce_id + ";" + ";".join(str(int(v)) for v in row) + "\n")

    forms_path = out_dir / "forms.csv"
    with forms_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("forme;frequencia;n_uce\n")
        for form in dtm.forms:
            handle.write(f"{form};{dtm.form_freq[form]};{dtm.form_nuce.get(form, 0)}\n")

    return {"dtm": dtm_path, "forms": forms_path}
