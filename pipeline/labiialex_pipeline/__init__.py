"""Pipeline labiia_lex: método Reinert/IRaMuTeQ aplicável no macOS (R + Python).

Parte 1 (fundação): importação de corpus, tokenização fiel ao IRaMuTeQ
(lematização + classificação ativa/suplementar via léxico), segmentação UCI/UCE
e construção da matriz UCE x formas ativas.
"""

from __future__ import annotations

from .corpus import (
    Uce,
    Uci,
    export_iramuteq_corpus,
    import_directory,
    segment_uces,
)
from .errors import (
    CorpusImportError,
    LexiqueError,
    MatrixError,
    PipelineError,
)
from .lexique import Lexique
from .matrix import DocTermMatrix, build_dtm, export_for_r
from .preprocess import PreprocessConfig, Preprocessor, build_preprocessor
from .tokenize import split_sentences, strip_accents, tokenize

__all__ = [
    "Uce",
    "Uci",
    "import_directory",
    "export_iramuteq_corpus",
    "segment_uces",
    "Lexique",
    "DocTermMatrix",
    "build_dtm",
    "export_for_r",
    "Preprocessor",
    "PreprocessConfig",
    "build_preprocessor",
    "tokenize",
    "split_sentences",
    "strip_accents",
    "PipelineError",
    "CorpusImportError",
    "LexiqueError",
    "MatrixError",
]
