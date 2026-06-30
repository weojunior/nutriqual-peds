"""Concordâncias KWIC (keyword in context) com casamento consistente.

Corrige a divergência do original (concordancer.py sem dobra de acentos vs
kwic_engine.py com dobra): aqui o casamento é por LEMA por padrão, sensível a
acento (preserva pares mínimos do pt-BR), com modos alternativos explícitos.
"""

from __future__ import annotations

from dataclasses import dataclass

from .corpus import Uci
from .lexique import Lexique
from .preprocess import Preprocessor
from .tokenize import tokenize

#: Janela de contexto (tokens à esquerda/direita) padrão.
DEFAULT_WINDOW: int = 6


@dataclass
class KwicLine:
    uci_id: str
    variables: dict[str, str]
    left: str
    keyword: str
    right: str
    position: int


def concordance(
    ucis: list[Uci],
    processor: "Preprocessor | Lexique",
    query: str,
    window: int = DEFAULT_WINDOW,
    mode: str = "lemma",
) -> list[KwicLine]:
    """Retorna linhas KWIC para ``query``.

    mode='lemma'  -> casa pelo lema (recomendado; agrupa flexões e sinônimos).
    mode='exact'  -> casa a forma exata (minúscula, com acento).
    """
    pre = processor if isinstance(processor, Preprocessor) else Preprocessor(processor)
    query_norm = query.lower().strip()
    target = pre.lemma(query_norm) if mode == "lemma" else query_norm

    lines: list[KwicLine] = []
    for uci in ucis:
        tokens = tokenize(uci.text)
        for pos, token in enumerate(tokens):
            key = pre.lemma(token) if mode == "lemma" else token
            if key != target:
                continue
            left = " ".join(tokens[max(0, pos - window): pos])
            right = " ".join(tokens[pos + 1: pos + 1 + window])
            lines.append(
                KwicLine(
                    uci_id=uci.uci_id, variables=dict(uci.variables),
                    left=left, keyword=token, right=right, position=pos,
                )
            )
    return lines
