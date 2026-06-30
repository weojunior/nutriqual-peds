"""Carregamento do léxico do IRaMuTeQ: lematização e classificação ativa/suplementar.

O IRaMuTeQ usa três recursos por idioma (diretório ``dictionaries``):

* ``lexique_<lang>.txt`` -- tab-separado, três colunas: ``forma  lema  tipo``.
* ``key.cfg`` -- mapeia cada ``tipo`` para 1 (forma ativa: substantivo, verbo,
  adjetivo, advérbio) ou 2 (forma suplementar: artigos, preposições, pronomes).
* ``expression_<lang>.txt`` -- tab-separado: ``expressão composta  forma_unica``.

A Classificação Hierárquica Descendente de Reinert e a AFC operam apenas sobre
as formas ATIVAS (chave 1), acima de uma frequência mínima.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

from .errors import LexiqueError

#: Chave atribuída a um tipo gramatical desconhecido no key.cfg.
#: Tipos não listados são tratados como ativos (chave 1), seguindo a convenção
#: de manter conteúdo lexical desconhecido (nomes próprios, neologismos).
DEFAULT_KEY_FOR_UNKNOWN_TYPE: int = 1

#: Tipo atribuído a uma forma ausente do léxico (mantida como ativa).
DEFAULT_TYPE_FOR_UNKNOWN_FORM: str = "nom"

#: Chave que identifica forma ativa.
ACTIVE_KEY: int = 1

#: Aliases de nomes de tipo divergentes entre lexique_*.txt e key.cfg.
#: (o léxico pt usa "conj" enquanto o key.cfg define "con").
_TYPE_ALIASES: dict[str, str] = {"conj": "con"}

#: Famílias de tipo gramatical consideradas ativas (conteúdo lexical).
_ACTIVE_PREFIXES: tuple[str, ...] = ("nom", "ver", "adj", "adv", "num")

#: Famílias de tipo gramatical consideradas suplementares (palavras-ferramenta).
_SUPPLEMENTARY_PREFIXES: tuple[str, ...] = ("art", "pre", "pro", "con", "aux", "ono")


@dataclass(frozen=True)
class LexEntry:
    """Resultado da consulta de uma forma no léxico."""

    forme: str
    lemme: str
    typ: str
    key: int

    @property
    def is_active(self) -> bool:
        return self.key == ACTIVE_KEY


class Lexique:
    """Léxico de um idioma carregado em memória."""

    def __init__(
        self,
        forms: dict[str, tuple[str, str]],
        keys: dict[str, int],
        expressions: dict[tuple[str, ...], str],
    ) -> None:
        self._forms = forms
        self._keys = keys
        self._expressions = expressions
        self._max_expr_len = max((len(k) for k in expressions), default=0)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, dictionaries_dir: Path, lang: str = "pt") -> "Lexique":
        """Carrega lexique_<lang>.txt, key.cfg e expression_<lang>.txt."""
        dictionaries_dir = Path(dictionaries_dir)
        lex_path = dictionaries_dir / f"lexique_{lang}.txt"
        key_path = dictionaries_dir / "key.cfg"
        expr_path = dictionaries_dir / f"expression_{lang}.txt"
        if not lex_path.exists():
            raise LexiqueError(f"Léxico não encontrado: {lex_path}")

        forms = cls._load_forms(lex_path)
        keys = cls._load_keys(key_path)
        expressions = cls._load_expressions(expr_path) if expr_path.exists() else {}
        return cls(forms, keys, expressions)

    @staticmethod
    def _load_forms(path: Path) -> dict[str, tuple[str, str]]:
        forms: dict[str, tuple[str, str]] = {}
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.rstrip("\n").rstrip("\r")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                forme, lemme, typ = parts[0], parts[1], parts[2]
                if forme:
                    forms[forme] = (lemme or forme, typ or DEFAULT_TYPE_FOR_UNKNOWN_FORM)
        if not forms:
            raise LexiqueError(f"Léxico vazio ou mal formatado: {path}")
        return forms

    @staticmethod
    def _load_keys(path: Path) -> dict[str, int]:
        if not path.exists():
            return {}
        parser = configparser.ConfigParser()
        parser.read(path, encoding="utf-8")
        keys: dict[str, int] = {}
        if parser.has_section("KEYS"):
            for typ, value in parser.items("KEYS"):
                try:
                    keys[typ.strip()] = int(value)
                except ValueError:
                    continue
        return keys

    @staticmethod
    def _load_expressions(path: Path) -> dict[tuple[str, ...], str]:
        expressions: dict[tuple[str, ...], str] = {}
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.rstrip("\n").rstrip("\r")
                if not line:
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) < 2:
                    continue
                phrase = tuple(parts[:-1]) if "\t" not in line else tuple(parts[0].split())
                replacement = parts[-1]
                if phrase and replacement:
                    expressions[tuple(t.lower() for t in phrase)] = replacement
        return expressions

    # --------------------------------------------------------------- queries
    def lookup(self, forme: str) -> LexEntry:
        """Retorna lema, tipo e chave de uma forma (sempre retorna algo)."""
        lemme, typ = self._forms.get(
            forme, (forme, DEFAULT_TYPE_FOR_UNKNOWN_FORM)
        )
        return LexEntry(forme=forme, lemme=lemme, typ=typ, key=self._resolve_key(typ))

    def _resolve_key(self, typ: str) -> int:
        """Resolve a chave (1 ativo / 2 suplementar) com fallback por família.

        Tipos ausentes do key.cfg são classificados por família gramatical,
        evitando que conjunções/preposições (ex.: 'conj' vs 'con') vazem como
        formas ativas e distorçam a CHD.
        """
        if typ in self._keys:
            return self._keys[typ]
        alias = _TYPE_ALIASES.get(typ)
        if alias and alias in self._keys:
            return self._keys[alias]
        if typ.endswith("_sup"):
            return 2
        if typ.startswith(_SUPPLEMENTARY_PREFIXES):
            return 2
        if typ.startswith(_ACTIVE_PREFIXES):
            return 1
        return DEFAULT_KEY_FOR_UNKNOWN_TYPE

    def is_known(self, forme: str) -> bool:
        """Indica se a forma consta do léxico (útil para política de desconhecidas)."""
        return forme in self._forms

    def apply_expressions(self, tokens: list[str]) -> list[str]:
        """Funde expressões compostas (greedy, casamento mais longo primeiro)."""
        if self._max_expr_len < 2:
            return tokens
        out: list[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            matched = False
            for length in range(min(self._max_expr_len, n - i), 1, -1):
                window = tuple(tokens[i : i + length])
                replacement = self._expressions.get(window)
                if replacement is not None:
                    out.append(replacement)
                    i += length
                    matched = True
                    break
            if not matched:
                out.append(tokens[i])
                i += 1
        return out

    @property
    def size(self) -> int:
        return len(self._forms)
