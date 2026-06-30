"""Análises complementares: nuvem de palavras, n-gramas, árvore de palavras,
heatmap e extração de termos (YAKE).

Operam sobre o corpus já importado (UCIs/UCEs) e/ou a matriz preparada.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .corpus import Uce
from .lexique import Lexique
from .preprocess import Preprocessor
from .tokenize import tokenize


# --------------------------------------------------------------- n-gramas
def ngram_counts(uces: list[Uce], n: int = 2) -> Counter:
    """Conta n-gramas de formas ATIVAS (na ordem do texto), por UCE."""
    counter: Counter = Counter()
    for uce in uces:
        seq = uce.active_lemmas
        for i in range(len(seq) - n + 1):
            counter[tuple(seq[i: i + n])] += 1
    return counter


# ------------------------------------------------------------- árvore de palavras
@dataclass
class WordTreeNode:
    word: str
    count: int
    children: dict = field(default_factory=dict)


def word_tree(
    ucis, processor: "Preprocessor | Lexique", query: str,
    direction: str = "right", depth: int = 2
) -> WordTreeNode:
    """Árvore de palavras: o que segue (right) ou precede (left) a palavra-chave."""
    pre = processor if isinstance(processor, Preprocessor) else Preprocessor(processor)
    target = pre.lemma(query.lower().strip())
    root = WordTreeNode(word=query, count=0)
    step = 1 if direction == "right" else -1
    for uci in ucis:
        tokens = tokenize(uci.text)
        lemmas = [pre.lemma(t) for t in tokens]
        for pos, lemma in enumerate(lemmas):
            if lemma != target:
                continue
            root.count += 1
            node = root
            for d in range(1, depth + 1):
                idx = pos + step * d
                if idx < 0 or idx >= len(tokens):
                    break
                w = tokens[idx]
                if w not in node.children:
                    node.children[w] = WordTreeNode(word=w, count=0)
                node.children[w].count += 1
                node = node.children[w]
    return root


def render_word_tree(node: WordTreeNode, indent: int = 0) -> list[str]:
    """Serializa a árvore em linhas de texto indentadas."""
    lines = []
    prefix = "  " * indent
    label = f"{node.word} ({node.count})" if indent else f"[{node.word}] ({node.count})"
    lines.append(prefix + label)
    for child in sorted(node.children.values(), key=lambda c: c.count, reverse=True):
        lines.extend(render_word_tree(child, indent + 1))
    return lines


# ----------------------------------------------------------------- heatmap
def form_class_proportions(
    dtm: np.ndarray, assignments: np.ndarray, forms: list[str],
    n_classes: int, top: int = 30
) -> tuple[np.ndarray, list[str], list[str]]:
    """Proporção de UCEs de cada classe que contêm cada forma (top formas)."""
    present = (dtm > 0).astype(float)
    freq = present.sum(axis=0)
    top_idx = np.argsort(freq)[::-1][:top]
    top_idx = top_idx[np.argsort([forms[i] for i in top_idx])]  # ordem alfabética
    rows = [forms[i] for i in top_idx]
    cols = [f"classe{c}" for c in range(1, n_classes + 1)]
    mat = np.zeros((len(rows), n_classes))
    for c in range(1, n_classes + 1):
        mask = assignments == c
        denom = max(int(mask.sum()), 1)
        mat[:, c - 1] = present[mask][:, top_idx].sum(axis=0) / denom
    return mat, rows, cols


# ------------------------------------------------------------------- YAKE
def yake_keyphrases(
    text: str, lang: str = "pt", n: int = 3, top: int = 20,
    stopwords: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Extrai palavras-chave/expressões com YAKE (menor score = mais relevante).

    ``stopwords`` (as palavras vazias do domínio) são unidas ao stopword interno
    do YAKE, para que a extração respeite a mesma limpeza das demais análises.
    """
    import yake

    extractor = yake.KeywordExtractor(lan=lang, n=n, top=top, dedupLim=0.9)
    if stopwords:
        extractor.stopword_set = extractor.stopword_set | {w.lower() for w in stopwords}
    return extractor.extract_keywords(text)
