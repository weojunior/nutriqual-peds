#!/usr/bin/env python3
"""Emoções NRC por falante (cuidador), a partir de transcrições com turnos rotulados.

Lê arquivos .docx cujos parágrafos começam com ``Participante N:`` (ou
``Palestrante N:``), concatena os turnos de cada participante e calcula as
emoções NRC por falante com o mesmo script R usado por documento. Os textos
por falante ficam num arquivo temporário apagado ao final. Só os agregados
são gravados: contagens por falante (sem texto) e número de falantes com pelo
menos uma ocorrência de cada emoção.

Uso:
    python run_emotions_por_falante.py --transcripts DIR --out DIR [--mode tokens|legacy]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

R_DIR = Path(__file__).resolve().parent / "r"
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
EMO_PT = {"anger": "raiva", "anticipation": "antecipação", "disgust": "nojo",
          "fear": "medo", "joy": "alegria", "sadness": "tristeza",
          "surprise": "surpresa", "trust": "confiança"}
ROTULOS_FALANTE = ("Participante", "Palestrante")
TURNO_RE = re.compile(
    rf"^\s*(?:{'|'.join(ROTULOS_FALANTE)})\s*(\d+)\s*:\s*(.*)$", re.S)
PARAGRAFO_RE = re.compile(r"<w:p[ >].*?</w:p>", re.S)
TAG_RE = re.compile(r"<[^>]+>")


class TranscricaoSemFalantesError(RuntimeError):
    """Nenhum turno rotulado por participante foi encontrado."""


def _paragrafos_docx(caminho: Path) -> list[str]:
    """Extrai o texto dos parágrafos de um .docx sem dependências externas."""
    xml = zipfile.ZipFile(caminho).read("word/document.xml").decode("utf-8")
    return [TAG_RE.sub("", p) for p in PARAGRAFO_RE.findall(xml)]


def falas_por_participante(caminho: Path) -> dict[int, list[str]]:
    """Agrupa os turnos de um arquivo por número de participante.

    Args:
        caminho: arquivo .docx com turnos ``Participante N: texto``.

    Returns:
        Dicionário participante -> lista de turnos, na ordem do arquivo.

    Raises:
        TranscricaoSemFalantesError: se nenhum turno rotulado for encontrado.
    """
    falas: dict[int, list[str]] = {}
    for p in _paragrafos_docx(caminho):
        m = TURNO_RE.match(p)
        if m:
            falas.setdefault(int(m.group(1)), []).append(m.group(2).strip())
    if not falas:
        raise TranscricaoSemFalantesError(f"sem turnos rotulados em {caminho.name}")
    return falas


def main() -> int:
    parser = argparse.ArgumentParser(description="Emoções NRC por falante")
    parser.add_argument("--transcripts", required=True, help="pasta com os .docx")
    parser.add_argument("--out", required=True, help="pasta de saída (só agregados)")
    parser.add_argument("--mode", choices=("tokens", "legacy"), default="tokens")
    args = parser.parse_args()

    pasta = Path(args.transcripts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    linhas = []
    for arq in sorted(pasta.glob("*.docx")):
        for pid, turnos in sorted(falas_por_participante(arq).items()):
            linhas.append({"doc_id": f"{arq.stem}_p{pid}", "grupo": arq.stem,
                           "participante": pid, "n_turnos": len(turnos),
                           "texto": " ".join(turnos).replace("\n", " ")})
    docs = pd.DataFrame(linhas)
    print(f"{len(docs)} falantes em {docs.grupo.nunique()} arquivos")

    with tempfile.TemporaryDirectory() as tmp:
        docs_path = Path(tmp) / "_docs_falantes.csv"
        docs.to_csv(docs_path, sep=";", index=False, encoding="utf-8")
        emo_path = Path(tmp) / "emo.csv"
        proc = subprocess.run(
            ["Rscript", str(R_DIR / "emotions_reference.R"), str(docs_path),
             str(emo_path), args.mode], capture_output=True, text=True)
        if not emo_path.exists():
            print("Falha no R:", proc.stderr[-600:])
            return 3
        emo = pd.read_csv(emo_path, sep=";")

    emo = emo.merge(docs[["doc_id", "grupo", "participante", "n_turnos"]], on="doc_id")
    emo.rename(columns=EMO_PT).to_csv(out_dir / "emotions_per_speaker.csv", sep=";",
                                      index=False, encoding="utf-8")
    n = len(emo)
    resumo = pd.DataFrame({
        "tokens_total": emo[EMOTIONS].sum().astype(int),
        "falantes_com_ocorrencia": (emo[EMOTIONS] > 0).sum().astype(int),
        "falantes_pct": (100 * (emo[EMOTIONS] > 0).sum() / n).round(1),
        "mediana_por_falante": emo[EMOTIONS].median(),
        "q1": emo[EMOTIONS].quantile(0.25),
        "q3": emo[EMOTIONS].quantile(0.75),
    }).rename(index=EMO_PT).sort_values("tokens_total", ascending=False)
    resumo.index.name = "emocao"
    resumo.to_csv(out_dir / "emotions_speakers_summary.csv", sep=";", encoding="utf-8")
    sem_emocao = int((emo[EMOTIONS].sum(axis=1) == 0).sum())
    print(f"modo {args.mode}: {int(emo[EMOTIONS].sum().sum())} ocorrências; "
          f"{sem_emocao} falante(s) sem ocorrência emocional")
    print(resumo.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
