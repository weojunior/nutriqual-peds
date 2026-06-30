"""Classificação Hierárquica Descendente de Reinert (núcleo canônico, em Python).

Reimplementação limpa e explicável do método do IRaMuTeQ (Pierre Ratinaud),
operando sobre a matriz binária UCE x forma ativa:

1. Extrai o primeiro fator de uma Análise Fatorial de Correspondência (AFC) da
   sub-matriz da classe a dividir.
2. Ordena as UCEs por essa coordenada.
3. Desliza o ponto de corte maximizando o qui-quadrado de inércia da bipartição
   (mesmo critério do ``MyChiSq``/``find.max`` do CHD.R).
4. Divide sempre a MAIOR classe (``which.max(tailleclasse)`` do CHD.R) até K classes.
5. Calcula as formas características de cada classe por qui-quadrado 2x2 assinado
   (df=1, limiar 3.84, sem correção de Yates -- convenção IRaMuTeQ).

Correções em relação ao código original: divisão da maior classe (e não da de
maior qui-quadrado), limiar fixo 3.84, sem Yates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

#: Tamanho mínimo de uma classe-filha para um corte ser aceito.
DEFAULT_MIN_CLASS_SIZE: int = 5

#: Limiar de qui-quadrado (df=1, p~0.05) para forma característica.
CHI2_THRESHOLD: float = 3.84

#: Frequência mínima de uma forma dentro da classe para entrar na AFC do corte.
MIN_COLSUM_IN_SUBSET: int = 2


@dataclass
class CharacteristicTerm:
    form: str
    chi2: float
    sign: str  # "+" (sobre-representada) ou "-" (sub-representada)
    n_in_class: int
    n_total: int
    min_expected: float = 0.0  # menor célula esperada da 2x2 (<5 => χ² frágil)


@dataclass
class ChdResult:
    n_classes: int
    assignments: np.ndarray  # classe (1..K) por UCE, na ordem de uce_ids
    uce_ids: list[str]
    class_sizes: dict[int, int]
    characteristic_terms: dict[int, list[CharacteristicTerm]]
    split_chi2: list[float] = field(default_factory=list)
    #: topologia da árvore descendente (dendrograma). Cada nó é um dict com
    #: ``children`` ([i, j] de índices em ``tree`` ou None p/ folha), ``chi2`` do
    #: corte, ``size`` (nº de UCEs), ``order`` (ordem da divisão) e, em folhas,
    #: ``label`` (classe final 1..K). A raiz é o nó 0.
    tree: list[dict] = field(default_factory=list)


def _ca_first_axis(sub: np.ndarray) -> np.ndarray:
    """Coordenada das linhas (UCEs) no 1.º fator não trivial da AFC.

    AFC clássica por SVD dos resíduos padronizados (Benzécri); a centragem
    ``P - r c^T`` já remove o eixo trivial, então o 1.º componente é o 1.º fator.
    """
    total = sub.sum()
    if total <= 0:
        return np.zeros(sub.shape[0])
    p = sub / total
    r = p.sum(axis=1)
    c = p.sum(axis=0)
    r_safe = np.where(r == 0, 1e-12, r)
    c_safe = np.where(c == 0, 1e-12, c)
    residual = (p - np.outer(r, c)) / np.sqrt(np.outer(r_safe, c_safe))
    try:
        u, s, _ = np.linalg.svd(residual, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(sub.shape[0])
    if s.size == 0:
        return np.zeros(sub.shape[0])
    # coordenada principal das linhas no 1.º eixo
    return (u[:, 0] * s[0]) / np.sqrt(r_safe)


def _best_inertia_cut(ordered: np.ndarray, min_size: int = 1) -> tuple[int, float]:
    """Corte que maximiza o qui-quadrado de inércia da bipartição 2xV.

    ``ordered`` é a sub-matriz (UCEs ordenadas pelo fator 1) x formas.
    Replica o critério de ``MyChiSq``/``find.max`` do CHD.R com somas cumulativas.
    Só considera cortes que deixam ambos os lados com pelo menos ``min_size`` UCEs.
    """
    col_tot = ordered.sum(axis=0).astype(float)  # totais de forma (V)
    n = col_tot.sum()
    if n <= 0:
        return 0, 0.0
    cum = np.cumsum(ordered, axis=0).astype(float)  # left = primeiras t UCEs
    n_rows = ordered.shape[0]
    lo = max(1, min_size)
    hi = n_rows - max(1, min_size)
    if hi < lo:                      # subgrupo pequeno: libera qualquer corte
        lo, hi = 1, n_rows - 1
    best_cut, best_chi2 = 0, -1.0
    for t in range(lo, hi + 1):  # corte entre a UCE t-1 e t
        left = cum[t - 1]
        right = col_tot - left
        sr = np.array([left.sum(), right.sum()])
        if sr[0] <= 0 or sr[1] <= 0:
            continue
        expected = np.outer(sr, col_tot) / n
        obs = np.vstack([left, right])
        mask = expected > 0
        chi2 = float(np.sum((obs[mask] - expected[mask]) ** 2 / expected[mask]))
        if chi2 > best_chi2:
            best_chi2, best_cut = chi2, t
    return best_cut, best_chi2


def _chi2_2xV(left: np.ndarray, right: np.ndarray, col_tot: np.ndarray, n: float) -> float:
    """Qui-quadrado de inércia de uma bipartição 2xV (mesmo critério do MyChiSq)."""
    sr = np.array([left.sum(), right.sum()])
    if sr[0] <= 0 or sr[1] <= 0 or n <= 0:
        return 0.0
    expected = np.outer(sr, col_tot) / n
    obs = np.vstack([left, right])
    mask = expected > 0
    return float(np.sum((obs[mask] - expected[mask]) ** 2 / expected[mask]))


def _relocate(
    sub: np.ndarray, left_mask: np.ndarray, min_class_size: int, max_passes: int = 30
) -> np.ndarray:
    """Fase de realocação do IRaMuTeQ: move UCEs entre as duas classes-filhas
    enquanto isso aumentar o qui-quadrado de inércia (refina o corte inicial)."""
    left_mask = left_mask.copy()
    col_tot = sub.sum(axis=0).astype(float)
    n = col_tot.sum()
    left_counts = sub[left_mask].sum(axis=0).astype(float)
    right_counts = col_tot - left_counts
    for _ in range(max_passes):
        changed = False
        cur = _chi2_2xV(left_counts, right_counts, col_tot, n)
        for i in range(sub.shape[0]):
            row = sub[i].astype(float)
            if left_mask[i]:
                if int(left_mask.sum()) - 1 < min_class_size:
                    continue
                new_left, new_right = left_counts - row, right_counts + row
            else:
                if int((~left_mask).sum()) - 1 < min_class_size:
                    continue
                new_left, new_right = left_counts + row, right_counts - row
            cand = _chi2_2xV(new_left, new_right, col_tot, n)
            if cand > cur + 1e-9:
                left_mask[i] = not left_mask[i]
                left_counts, right_counts = new_left, new_right
                cur = cand
                changed = True
        if not changed:
            break
    return left_mask


def _signed_chi2_terms(
    matrix: np.ndarray,
    in_class: np.ndarray,
    forms: list[str],
    threshold: float,
) -> list[CharacteristicTerm]:
    """Formas características de uma classe por qui-quadrado 2x2 assinado (df=1)."""
    n_total = matrix.shape[0]
    n_in = int(in_class.sum())
    n_out = n_total - n_in
    if n_in == 0 or n_out == 0:
        return []
    present = matrix > 0
    a = present[in_class].sum(axis=0).astype(float)          # presente & na classe
    b = present[~in_class].sum(axis=0).astype(float)         # presente & fora
    c = n_in - a                                             # ausente & na classe
    d = n_out - b                                            # ausente & fora
    n = float(n_total)
    # qui-quadrado 2x2 sem Yates: n(ad-bc)^2 / ((a+b)(c+d)(a+c)(b+d))
    num = n * (a * d - b * c) ** 2
    den = (a + b) * (c + d) * (a + c) * (b + d)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(den > 0, num / den, 0.0)
    expected_present = (a + b) * n_in / n
    # menor célula esperada da tabela 2x2 (membership x presença da forma)
    min_exp = np.minimum.reduce([
        (a + b) * n_in / n, (a + b) * n_out / n,
        (c + d) * n_in / n, (c + d) * n_out / n,
    ])
    out: list[CharacteristicTerm] = []
    for j, form in enumerate(forms):
        if chi2[j] >= threshold and a[j] > 0:
            sign = "+" if a[j] >= expected_present[j] else "-"
            out.append(
                CharacteristicTerm(
                    form=form, chi2=float(chi2[j]), sign=sign,
                    n_in_class=int(a[j]), n_total=int(a[j] + b[j]),
                    min_expected=round(float(min_exp[j]), 2),
                )
            )
    out.sort(key=lambda t: t.chi2, reverse=True)
    return out


def run_chd(
    dtm: sparse.csr_matrix,
    forms: list[str],
    uce_ids: list[str],
    n_classes: int = 5,
    min_class_size: int = DEFAULT_MIN_CLASS_SIZE,
    chi2_threshold: float = CHI2_THRESHOLD,
    relocate: bool = True,
) -> ChdResult:
    """Executa a CHD de Reinert e retorna classes + formas características."""
    matrix = np.asarray(dtm.todense(), dtype=float)
    n_uce = matrix.shape[0]
    # cada classe é o conjunto de índices de linha; começa com tudo na classe 1
    classes: dict[int, np.ndarray] = {1: np.arange(n_uce)}
    next_id = 2
    frozen: set[int] = set()
    split_chi2: list[float] = []
    # árvore do dendrograma: nó 0 = raiz (todas as UCEs)
    tree: list[dict] = [{"children": None, "chi2": None, "size": n_uce, "order": 0}]
    node_of: dict[int, int] = {1: 0}

    while len(classes) < n_classes:
        # candidata: a MAIOR classe ainda divisível (CHD.R: which.max), excluindo
        # as marcadas como não divisíveis (evita laço infinito em dados reais)
        candidates = {
            cid: rows for cid, rows in classes.items()
            if cid not in frozen and len(rows) >= 2 * min_class_size
        }
        if not candidates:
            break
        cid = max(candidates, key=lambda c: len(candidates[c]))
        rows = classes[cid]
        sub_full = matrix[rows]
        keep = sub_full.sum(axis=0) >= MIN_COLSUM_IN_SUBSET
        sub = sub_full[:, keep] if keep.any() else sub_full
        axis1 = _ca_first_axis(sub)
        order = np.argsort(axis1, kind="stable")
        ordered = sub[order]
        cut, chi2 = _best_inertia_cut(ordered, min_class_size)
        if cut < min_class_size or (len(rows) - cut) < min_class_size:
            # nenhum corte respeita o tamanho mínimo; marca como não divisível
            frozen.add(cid)
            continue
        mask = np.zeros(sub.shape[0], dtype=bool)
        mask[order[:cut]] = True
        if relocate:
            mask = _relocate(sub, mask, min_class_size)
        if int(mask.sum()) < min_class_size or int((~mask).sum()) < min_class_size:
            frozen.add(cid)
            continue
        col_tot = sub.sum(axis=0).astype(float)
        left_counts = sub[mask].sum(axis=0).astype(float)
        chi2 = _chi2_2xV(left_counts, col_tot - left_counts, col_tot, col_tot.sum())
        left_local = np.where(mask)[0]
        right_local = np.where(~mask)[0]
        classes.pop(cid)
        classes[cid] = rows[left_local]
        classes[next_id] = rows[right_local]
        split_chi2.append(chi2)
        # registra o nó interno e os dois filhos no dendrograma
        parent_node = node_of[cid]
        left_node, right_node = len(tree), len(tree) + 1
        tree.append({"children": None, "chi2": None, "size": int(left_local.size),
                     "order": len(split_chi2)})
        tree.append({"children": None, "chi2": None, "size": int(right_local.size),
                     "order": len(split_chi2)})
        tree[parent_node].update(children=[left_node, right_node], chi2=float(chi2),
                                 order=len(split_chi2))
        node_of[cid] = left_node
        node_of[next_id] = right_node
        next_id += 1

    # renomeia classes para 1..K em ordem de tamanho decrescente
    final = {k: v for k, v in classes.items()}
    ordered_ids = sorted(final, key=lambda c: len(final[c]), reverse=True)
    assignments = np.zeros(n_uce, dtype=int)
    relabel: dict[int, int] = {}
    for new_id, old in enumerate(ordered_ids, start=1):
        relabel[new_id] = old
        for idx in final[old]:
            assignments[idx] = new_id

    characteristic: dict[int, list[CharacteristicTerm]] = {}
    class_sizes: dict[int, int] = {}
    for new_id in range(1, len(ordered_ids) + 1):
        in_class = assignments == new_id
        class_sizes[new_id] = int(in_class.sum())
        characteristic[new_id] = _signed_chi2_terms(
            matrix, in_class, forms, chi2_threshold
        )
    # rotula as folhas do dendrograma com a classe final (1..K)
    for new_id, old in relabel.items():
        leaf = node_of.get(old)
        if leaf is not None and tree[leaf]["children"] is None:
            tree[leaf]["label"] = new_id
    return ChdResult(
        n_classes=len(ordered_ids),
        assignments=assignments,
        uce_ids=uce_ids,
        class_sizes=class_sizes,
        characteristic_terms=characteristic,
        split_chi2=split_chi2,
        tree=tree,
    )


def chd_stability(
    dtm: sparse.csr_matrix,
    forms: list[str],
    uce_ids: list[str],
    assignments: np.ndarray,
    n_classes: int,
    min_class_size: int = DEFAULT_MIN_CLASS_SIZE,
    n_runs: int = 8,
    subsample: float = 0.85,
    seed: int = 0,
) -> dict:
    """Estabilidade da CHD por subamostragem: co-classificação de pares de UCEs.

    Roda a CHD em ``n_runs`` subamostras e mede com que frequência pares de UCEs
    da MESMA classe de referência continuam juntos (within) versus pares de
    classes diferentes (between). within alto e between baixo => classes estáveis.
    """
    rng = np.random.RandomState(seed)
    n = dtm.shape[0]
    coassoc = np.zeros((n, n), dtype=np.float32)
    copresent = np.zeros((n, n), dtype=np.float32)
    size = max(int(subsample * n), 2 * min_class_size + 1)
    for r in range(n_runs):
        idx = np.sort(rng.choice(n, size=min(size, n), replace=False))
        res = run_chd(dtm[idx], forms, [uce_ids[i] for i in idx],
                      n_classes=n_classes, min_class_size=min_class_size, relocate=True)
        present = np.zeros(n, dtype=bool)
        present[idx] = True
        labels = np.full(n, -1)
        labels[idx] = res.assignments
        po = np.outer(present, present)
        copresent += po
        coassoc += ((labels[:, None] == labels[None, :]) & po).astype(np.float32)

    triu = np.triu_indices(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(copresent > 0, coassoc / np.maximum(copresent, 1), np.nan)
    same = assignments[:, None] == assignments[None, :]
    within = float(np.nanmean(rates[triu][same[triu]]))
    between = float(np.nanmean(rates[triu][~same[triu]]))
    return {
        "n_runs": n_runs,
        "subsample": subsample,
        "within_class_coassoc": round(within, 4),
        "between_class_coassoc": round(between, 4),
        "stability": round(within - between, 4),
    }
