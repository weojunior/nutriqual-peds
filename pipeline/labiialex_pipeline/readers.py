"""Leitores de corpus em múltiplos formatos (.txt, .md, .pdf, .docx, .odt, .csv, .xlsx, .zip).

Formatos de um documento por arquivo retornam texto; planilhas (.csv/.xlsx)
expandem uma linha por documento, com as demais colunas viram variáveis.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from .errors import CorpusImportError

#: Extensões em que cada arquivo é UM documento.
SINGLE_DOC_SUFFIXES: tuple[str, ...] = (".txt", ".text", ".md", ".markdown", ".pdf", ".docx", ".odt")
#: Extensões de planilha (uma linha = um documento).
SPREADSHEET_SUFFIXES: tuple[str, ...] = (".csv", ".tsv", ".xlsx", ".xls")
#: Extensões compactadas.
ARCHIVE_SUFFIXES: tuple[str, ...] = (".zip",)

#: Codificações tentadas em arquivos de texto.
_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

#: Nomes prováveis de coluna de texto em planilhas (minúsculo).
_TEXT_COLUMN_HINTS = ("texto", "text", "conteudo", "conteúdo", "transcricao",
                      "transcrição", "fala", "resposta", "body", "content")


#: Falantes descartados por padrão (moldura do grupo, não discurso dos participantes).
DEFAULT_DROP_SPEAKERS: tuple[str, ...] = (
    "moderador", "mediador", "palestrante", "entrevistador",
    "facilitador", "pesquisador", "coordenador", "apresentador",
)

#: Rótulo de falante no início de um parágrafo ("Participante 6:", "Moderador:", ...).
_SPEAKER_RE = re.compile(r"^\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ.\d ]{0,38}?)\s*:\s", re.UNICODE)


#: Anotações do transcritor entre parênteses ou colchetes (ex.: "(inaudível)").
_ARTIFACT_BRACKETS = re.compile(r"[\(\[][^)\]\n]{0,60}[\)\]]")
#: Marcações soltas que nunca são conteúdo.
_ARTIFACT_WORDS = re.compile(r"(?i)\b(?:inaud[ií]vel|incompreens[ií]vel|ininteleg[ií]vel|ininteligível)\b")
#: Rótulos de fala residuais dentro do texto (ex.: "Participante 3", "Moderador").
_LABEL_WORDS = re.compile(
    r"(?i)\b(?:participante|moderador|palestrante|entrevistador[a]?|mediador[a]?)\s*\d*\b"
)


def clean_transcription_artifacts(text: str) -> str:
    """Remove anotações de transcrição e rótulos de fala residuais do texto."""
    text = _ARTIFACT_BRACKETS.sub(" ", text)
    text = _ARTIFACT_WORDS.sub(" ", text)
    text = _LABEL_WORDS.sub(" ", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def split_turns(text: str) -> list[tuple[str | None, str]]:
    """Divide um texto em turnos (falante, fala) pelos rótulos no início de linha."""
    turns: list[tuple[str | None, str]] = []
    speaker: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        match = _SPEAKER_RE.match(line)
        if match:
            if buf or speaker is not None:
                turns.append((speaker, "\n".join(buf)))
            speaker = match.group(1).strip()
            buf = [line[match.end():]]
        else:
            buf.append(line)
    if buf or speaker is not None:
        turns.append((speaker, "\n".join(buf)))
    return turns


def filter_speaker_turns(
    text: str, drop_patterns: list[str] | tuple[str, ...] | None
) -> tuple[str, int]:
    """Remove turnos cujo falante casa com ``drop_patterns``; retorna (texto, n_removidos).

    Se nenhum falante for detectado no texto, devolve o texto original intacto.
    """
    if not drop_patterns:
        return text, 0
    drop_re = re.compile(r"(?i)^\s*(?:" + "|".join(drop_patterns) + r")")
    turns = split_turns(text)
    if not any(spk for spk, _ in turns):
        return text, 0
    kept, dropped = [], 0
    for spk, body in turns:
        if spk and drop_re.match(spk):
            dropped += 1
        else:
            kept.append(body)
    return "\n".join(kept), dropped


def _read_plain(path: Path) -> str:
    for encoding in _ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise CorpusImportError(f"Não foi possível decodificar {path}")


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)        # blocos de código
    text = re.sub(r"`[^`]*`", " ", text)                           # código inline
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)              # imagens
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)           # links -> texto
    text = re.sub(r"^[#>\-\*\+\s]+", "", text, flags=re.MULTILINE)  # marcadores
    text = re.sub(r"[*_~#]+", "", text)                            # ênfase
    return text


def _read_pdf(path: Path) -> str:
    from pdfminer.high_level import extract_text
    return extract_text(str(path)) or ""


def _read_docx(path: Path) -> str:
    import docx
    return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)


def _read_odt(path: Path) -> str:
    from odf.opendocument import load
    from odf.text import P
    from odf import teletype
    doc = load(str(path))
    return "\n".join(teletype.extractText(p) for p in doc.getElementsByType(P))


def read_single_document(path: Path) -> str:
    """Lê um arquivo de documento único e devolve o texto."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".text"):
        return _read_plain(path)
    if suffix in (".md", ".markdown"):
        return _strip_markdown(_read_plain(path))
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".odt":
        return _read_odt(path)
    raise CorpusImportError(f"Formato de documento não suportado: {path}")


def _detect_text_column(frame) -> str:
    lower = {c.lower(): c for c in frame.columns}
    for hint in _TEXT_COLUMN_HINTS:
        if hint in lower:
            return lower[hint]
    # heurística: coluna textual com maior comprimento médio
    best, best_len = None, -1.0
    for col in frame.columns:
        series = frame[col].astype(str)
        avg = series.str.len().mean()
        if avg > best_len:
            best, best_len = col, avg
    return best


def read_spreadsheet(
    path: Path, text_column: str | None = None
) -> list[tuple[str, dict[str, str], str]]:
    """Lê uma planilha: cada linha vira (doc_id, variáveis, texto)."""
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        frame = pd.read_excel(path, dtype=str)
    else:
        sep = "\t" if suffix == ".tsv" else None
        frame = pd.read_csv(path, dtype=str, sep=sep, engine="python", encoding="utf-8-sig")
    frame = frame.fillna("")
    if frame.empty:
        raise CorpusImportError(f"Planilha vazia: {path}")

    text_col = text_column or _detect_text_column(frame)
    if text_col not in frame.columns:
        raise CorpusImportError(f"Coluna de texto '{text_col}' não existe em {path}")

    id_col = next((c for c in frame.columns if c.lower() in ("id", "file", "doc_id", "documento")), None)
    var_cols = [c for c in frame.columns if c not in (text_col, id_col)]
    out: list[tuple[str, dict[str, str], str]] = []
    for i, row in frame.iterrows():
        doc_id = str(row[id_col]) if id_col else f"{path.stem}_lin{i + 1:04d}"
        variables = {c: str(row[c]) for c in var_cols}
        out.append((doc_id, variables, str(row[text_col])))
    return out


def expand_archive(path: Path, dest_dir: Path) -> list[Path]:
    """Extrai um .zip e devolve os arquivos suportados de dentro dele."""
    extracted: list[Path] = []
    with zipfile.ZipFile(path) as archive:
        archive.extractall(dest_dir)
    supported = SINGLE_DOC_SUFFIXES + SPREADSHEET_SUFFIXES
    for item in sorted(dest_dir.rglob("*")):
        if item.is_file() and item.suffix.lower() in supported and item.name != "metadata.csv":
            extracted.append(item)
    return extracted
