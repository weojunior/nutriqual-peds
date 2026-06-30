"""Exceções customizadas do pipeline labiia_lex (método Reinert/IRaMuTeQ)."""

from __future__ import annotations


class PipelineError(Exception):
    """Erro base do pipeline."""


class CorpusImportError(PipelineError):
    """Falha ao importar ou ler arquivos do corpus."""


class LexiqueError(PipelineError):
    """Falha ao carregar o léxico de lematização do IRaMuTeQ."""


class MatrixError(PipelineError):
    """Falha ao construir a matriz UCE x formas (corpus pequeno demais, etc.)."""
