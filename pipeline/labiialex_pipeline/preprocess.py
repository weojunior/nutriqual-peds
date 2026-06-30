"""Pré-processamento configurável (ajuste ao domínio antes da matriz).

Permite controlar, sem editar código:

* stopwords adicionais (formas/lemmas forçados a suplementar);
* expressões multipalavra a preservar como uma forma (ex.: "dor abdominal");
* sinônimos/variantes a unificar (ex.: "medicamento" -> "remédio");
* advérbios como ativos ou suplementares;
* formas desconhecidas (fora do léxico) como ativas ou suplementares;
* tamanho mínimo de token.

O ``Preprocessor`` encapsula o léxico do IRaMuTeQ mais essa configuração e é
usado para extrair as formas ativas de cada UCE.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .lexique import Lexique
from .tokenize import tokenize


@dataclass
class PreprocessConfig:
    stopwords: frozenset[str] = frozenset()
    synonyms: dict[str, str] = field(default_factory=dict)
    user_expressions: dict[tuple[str, ...], str] = field(default_factory=dict)
    adverbs_active: bool = True
    unknown_active: bool = True
    min_token_len: int = 1
    anon_prefix: str | None = None  # tokens com este prefixo (marcadores) viram suplementares

    # ---------------------------------------------------------- carregadores
    @staticmethod
    def load_stopwords(path: Path) -> frozenset[str]:
        words = set()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            w = line.strip().lower()
            if w and not w.startswith("#"):
                words.add(w)
        return frozenset(words)

    @staticmethod
    def load_synonyms(path: Path) -> dict[str, str]:
        """CSV/TSV com colunas 'variante' e 'canonico' (ou duas colunas)."""
        out: dict[str, str] = {}
        text = Path(path).read_text(encoding="utf-8")
        delimiter = "\t" if "\t" in text.splitlines()[0] else ","
        reader = csv.reader(text.splitlines(), delimiter=delimiter)
        rows = list(reader)
        header = [h.strip().lower() for h in rows[0]] if rows else []
        start = 1 if set(header) & {"variante", "canonico", "canônico", "from", "to"} else 0
        for row in rows[start:]:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                out[row[0].strip().lower()] = row[1].strip().lower()
        return out

    @staticmethod
    def load_expressions(path: Path) -> dict[tuple[str, ...], str]:
        """Uma expressão por linha ('dor abdominal'); vira token 'dor_abdominal'."""
        out: dict[tuple[str, ...], str] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip().lower()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                phrase, repl = line.split("\t", 1)
                parts = tuple(phrase.split())
                replacement = repl.strip()
            else:
                parts = tuple(line.split())
                replacement = "_".join(parts)
            if len(parts) >= 2:
                out[parts] = replacement
        return out


class Preprocessor:
    """Aplica léxico + configuração para obter as formas ativas de um texto."""

    def __init__(self, lexique: Lexique, config: PreprocessConfig | None = None) -> None:
        self.lex = lexique
        self.cfg = config or PreprocessConfig()
        self._expr_values = set(self.cfg.user_expressions.values())
        self._max_user_expr = max((len(k) for k in self.cfg.user_expressions), default=0)
        self._syn = self._resolve_synonym_chains(self.cfg.synonyms)

    @staticmethod
    def _resolve_synonym_chains(syn: dict[str, str]) -> dict[str, str]:
        """Resolve cadeias de sinônimos (a->b->c vira a->c), com guarda de ciclo."""
        resolved: dict[str, str] = {}
        for key in syn:
            seen, cur = set(), key
            while cur in syn and cur not in seen:
                seen.add(cur)
                cur = syn[cur]
            resolved[key] = cur
        return resolved

    # ------------------------------------------------------------ operações
    def apply_expressions(self, tokens: list[str]) -> list[str]:
        """Funde expressões do usuário (primeiro) e depois as do léxico."""
        tokens = self._apply_user_expressions(tokens)
        return self.lex.apply_expressions(tokens)

    def _apply_user_expressions(self, tokens: list[str]) -> list[str]:
        if self._max_user_expr < 2:
            return tokens
        out: list[str] = []
        i, n = 0, len(tokens)
        while i < n:
            matched = False
            for length in range(min(self._max_user_expr, n - i), 1, -1):
                window = tuple(tokens[i: i + length])
                repl = self.cfg.user_expressions.get(window)
                if repl is not None:
                    out.append(repl)
                    i += length
                    matched = True
                    break
            if not matched:
                out.append(tokens[i])
                i += 1
        return out

    def lemma(self, token: str) -> str:
        lem = self.lex.lookup(token).lemme
        if lem in self._syn:
            return self._syn[lem]
        if token in self._syn:
            return self._syn[token]
        return lem

    def is_active(self, token: str) -> bool:
        if self.cfg.anon_prefix and token.startswith(self.cfg.anon_prefix):
            return False                       # marcador de anonimização
        if token in self._expr_values:        # expressão preservada = conteúdo
            return True
        if len(token) < self.cfg.min_token_len:
            return False
        entry = self.lex.lookup(token)
        lem = self.lemma(token)
        if token in self.cfg.stopwords or lem in self.cfg.stopwords:
            return False
        if not self.cfg.adverbs_active and entry.typ.startswith("adv"):
            return False
        if not self.cfg.unknown_active and not self.lex.is_known(token):
            return False
        return entry.is_active

    def active_lemmas(self, text: str) -> list[str]:
        tokens = self.apply_expressions(tokenize(text))
        return [self.lemma(t) for t in tokens if self.is_active(t)]


def build_preprocessor(
    lexique: Lexique,
    stopwords_path: str | Path | None = None,
    synonyms_path: str | Path | None = None,
    expressions_path: str | Path | None = None,
    adverbs_active: bool = True,
    unknown_active: bool = True,
    min_token_len: int = 1,
    anon_prefix: str | None = None,
) -> Preprocessor:
    """Conveniência: monta um Preprocessor a partir de arquivos de configuração."""
    cfg = PreprocessConfig(
        stopwords=PreprocessConfig.load_stopwords(stopwords_path) if stopwords_path else frozenset(),
        synonyms=PreprocessConfig.load_synonyms(synonyms_path) if synonyms_path else {},
        user_expressions=PreprocessConfig.load_expressions(expressions_path) if expressions_path else {},
        adverbs_active=adverbs_active,
        unknown_active=unknown_active,
        min_token_len=min_token_len,
        anon_prefix=anon_prefix,
    )
    return Preprocessor(lexique, cfg)


def load_preprocessor(info: dict, lexique: Lexique) -> Preprocessor:
    """Reconstrói o Preprocessor a partir do bloco 'preprocess' do prepared.json.

    Garante que KWIC, sentimento e LDA usem o MESMO pré-processamento da matriz.
    """
    cfg = info.get("preprocess", {}) or {}
    return build_preprocessor(
        lexique,
        stopwords_path=cfg.get("stopwords"),
        synonyms_path=cfg.get("synonyms"),
        expressions_path=cfg.get("expressions"),
        adverbs_active=cfg.get("adverbs_active", True),
        unknown_active=cfg.get("unknown_active", True),
        min_token_len=cfg.get("min_token_len", 1),
        anon_prefix=cfg.get("anon_prefix"),
    )
