"""Tokenizador único, Unicode-aware (correção do bug de tokenização original).

O código original usava várias expressões divergentes como ``[a-zA-ZÀ-ÿ]+``, que
incluem os sinais × (U+00D7) e ÷ (U+00F7) como se fossem letras e ignoram letras
fora do Latin-1. Aqui usamos a classe Unicode de letras, garantindo contagem
consistente entre frequência, KWIC e especificidades.
"""

from __future__ import annotations

import re
import unicodedata

#: Sequências de uma ou mais letras Unicode (sem dígitos, sem sublinhado).
#: ``[^\W\d_]`` = caractere de palavra que não é dígito nem sublinhado => letra.
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

#: Sentenças: corta em pontuação final ou quebras de linha.
_SENTENCE_RE = re.compile(r"[^.!?;\n\r]+[.!?;]?", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Quebra o texto em tokens minúsculos de letras, preservando acentos."""
    return _WORD_RE.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    """Segmenta em sentenças por pontuação final e quebras de linha."""
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


def strip_accents(text: str) -> str:
    """Remove acentos (uso OPCIONAL; desligado por padrão para preservar pt-BR).

    Manter desligado evita fundir pares mínimos do português como
    está/esta, é/e, só/so, pôde/pode.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))
