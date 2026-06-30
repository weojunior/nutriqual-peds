"""Importação do corpus e segmentação UCI/UCE (terminologia IRaMuTeQ).

* UCI  -- unidade de contexto inicial: cada arquivo (uma entrevista/grupo focal).
* UCE  -- unidade de contexto elementar: segmento de ~``uce_size`` tokens,
  respeitando limites de sentença. É a unidade de análise da CHD de Reinert.

Metadados (variáveis) por arquivo: opcional ``metadata.csv`` no diretório do
corpus, com uma coluna ``file`` (nome do arquivo) e demais colunas como
variáveis (ex.: grupo, sexo, idade). Sem ele, cria-se a variável ``source``.
"""

from __future__ import annotations

import csv
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .errors import CorpusImportError
from .lexique import Lexique
from .preprocess import Preprocessor
from .readers import (
    ARCHIVE_SUFFIXES,
    SINGLE_DOC_SUFFIXES,
    SPREADSHEET_SUFFIXES,
    clean_transcription_artifacts,
    expand_archive,
    filter_speaker_turns,
    read_single_document,
    read_spreadsheet,
)
from .tokenize import split_sentences, tokenize

#: Tamanho-alvo de uma UCE em número de tokens (padrão do IRaMuTeQ/config).
DEFAULT_UCE_SIZE: int = 40

#: Extensões de arquivo aceitas como documentos do corpus.
TEXT_SUFFIXES: tuple[str, ...] = (".txt", ".text")
DOCX_SUFFIXES: tuple[str, ...] = (".docx",)

#: Codificações tentadas ao ler .txt, em ordem.
_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


@dataclass
class Uce:
    """Segmento elementar de contexto."""

    uci_id: str
    index: int
    text: str
    tokens: list[str] = field(default_factory=list)
    active_lemmas: list[str] = field(default_factory=list)

    @property
    def uce_id(self) -> str:
        return f"{self.uci_id}__uce{self.index:04d}"


@dataclass
class Uci:
    """Documento inteiro (uma entrevista/grupo) com suas variáveis."""

    uci_id: str
    variables: dict[str, str]
    text: str
    uces: list[Uce] = field(default_factory=list)


def _load_metadata(corpus_dir: Path) -> dict[str, dict[str, str]]:
    """Lê metadata.csv tolerando o que apps de planilha costumam fazer:
    separador vírgula ou ponto e vírgula e linhas-lixo antes do cabeçalho."""
    meta_path = corpus_dir / "metadata.csv"
    if not meta_path.exists():
        return {}
    text = meta_path.read_text(encoding="utf-8-sig")
    delimiter = ";" if text.count(";") >= text.count(",") else ","
    rows = [r for r in csv.reader(text.splitlines(), delimiter=delimiter)]
    header_idx = next(
        (i for i, r in enumerate(rows) if any(c.strip().lower() == "file" for c in r)),
        None,
    )
    if header_idx is None:
        raise CorpusImportError("metadata.csv precisa de uma coluna 'file'")
    header = [c.strip() for c in rows[header_idx]]
    file_col = next(j for j, c in enumerate(header) if c.lower() == "file")
    out: dict[str, dict[str, str]] = {}
    for r in rows[header_idx + 1:]:
        if len(r) <= file_col:
            continue
        key = r[file_col].strip()
        if not key:
            continue
        out[key] = {
            header[j]: r[j].strip()
            for j in range(len(header))
            if j != file_col and j < len(r) and header[j]
        }
    return out


def import_directory(
    corpus_dir: str | Path,
    text_column: str | None = None,
    drop_speakers: list[str] | tuple[str, ...] | None = None,
) -> list[Uci]:
    """Importa um diretório de corpus em múltiplos formatos e retorna as UCIs.

    Cada .txt/.md/.pdf/.docx/.odt vira uma UCI; cada linha de planilha
    (.csv/.xlsx/.tsv) vira uma UCI; arquivos .zip são extraídos e seu conteúdo
    incorporado. O ``metadata.csv`` (se houver) atribui variáveis aos arquivos
    de documento único pelo nome de arquivo.

    ``drop_speakers``: se informado, remove dos documentos de texto os turnos
    cujo falante casa com esses padrões (ex.: moderador, palestrante).
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise CorpusImportError(f"Diretório de corpus inexistente: {corpus_dir}")

    metadata = _load_metadata(corpus_dir)
    # ignora qualquer arquivo "metadata*" (config, não documento do corpus)
    top_files = sorted(
        p for p in corpus_dir.iterdir()
        if p.is_file() and not p.name.lower().startswith("metadata")
    )

    worklist: list[Path] = []
    tmp_dirs: list[Path] = []
    for path in top_files:
        if path.suffix.lower() in ARCHIVE_SUFFIXES:
            tmp = Path(tempfile.mkdtemp(prefix="labiialex_zip_"))
            tmp_dirs.append(tmp)
            worklist.extend(expand_archive(path, tmp))
        else:
            worklist.append(path)

    ucis: list[Uci] = []
    try:
        for path in worklist:
            suffix = path.suffix.lower()
            if suffix in SPREADSHEET_SUFFIXES:
                for doc_id, variables, text in read_spreadsheet(path, text_column):
                    ucis.append(Uci(uci_id=doc_id,
                                    variables=variables or {"source": doc_id},
                                    text=text))
            elif suffix in SINGLE_DOC_SUFFIXES:
                text = read_single_document(path)
                if drop_speakers:
                    text, _ = filter_speaker_turns(text, drop_speakers)
                text = clean_transcription_artifacts(text)
                variables = dict(metadata.get(path.name, {})) or {"source": path.stem}
                ucis.append(Uci(uci_id=path.stem, variables=variables, text=text))
    finally:
        for tmp in tmp_dirs:
            shutil.rmtree(tmp, ignore_errors=True)

    if not ucis:
        raise CorpusImportError(
            f"Nenhum documento suportado em {corpus_dir} "
            "(.txt/.md/.pdf/.docx/.odt/.csv/.xlsx/.zip)."
        )
    return ucis


def export_iramuteq_corpus(ucis: list[Uci], path: str | Path) -> Path:
    """Exporta o corpus limpo no formato de texto do IRaMuTeQ (linhas ****)."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for uci in ucis:
            stars = " ".join(
                f"*{key}_{str(value).strip().replace(' ', '-')}"
                for key, value in uci.variables.items() if str(value).strip()
            )
            handle.write(f"**** {stars}\n".rstrip() + "\n")
            handle.write(uci.text.strip() + "\n\n")
    return path


def segment_uces(
    uci: Uci, processor: "Preprocessor | Lexique", uce_size: int = DEFAULT_UCE_SIZE
) -> list[Uce]:
    """Segmenta uma UCI em UCEs de ~``uce_size`` tokens (limites de sentença).

    ``processor`` pode ser um ``Preprocessor`` (com configuração de domínio) ou
    um ``Lexique`` simples (configuração padrão, comportamento original).
    """
    pre = processor if isinstance(processor, Preprocessor) else Preprocessor(processor)
    sentences = split_sentences(uci.text)
    uces: list[Uce] = []
    buffer_sentences: list[str] = []
    buffer_count = 0
    index = 0

    def flush() -> None:
        nonlocal buffer_sentences, buffer_count, index
        if not buffer_sentences:
            return
        seg_text = " ".join(buffer_sentences)
        tokens = pre.apply_expressions(tokenize(seg_text))
        active = [pre.lemma(tok) for tok in tokens if pre.is_active(tok)]
        uces.append(
            Uce(
                uci_id=uci.uci_id,
                index=index,
                text=seg_text,
                tokens=tokens,
                active_lemmas=active,
            )
        )
        index += 1
        buffer_sentences = []
        buffer_count = 0

    for sentence in sentences:
        n_tokens = len(tokenize(sentence))
        buffer_sentences.append(sentence)
        buffer_count += n_tokens
        if buffer_count >= uce_size:
            flush()
    flush()
    uci.uces = uces
    return uces
