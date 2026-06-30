#!/usr/bin/env python3
"""CLI da Parte 1: prepara um corpus (um arquivo por entrevista/grupo) para análise.

Uso:
    python run_prepare.py --corpus CORPUS_DIR --out OUT_DIR \
        [--lang pt] [--uce-size 40] [--min-freq 3] [--dictionaries DIR]

Saídas em OUT_DIR:
    dtm.csv         matriz binária UCE x forma ativa (para os scripts R)
    forms.csv       formas ativas retidas (frequência, n_uce)
    uce_meta.csv    metadados por UCE (UCI de origem, variáveis, contagens)
    prepared.json   resumo da preparação
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from labiialex_pipeline import (
    Lexique,
    build_dtm,
    build_preprocessor,
    export_for_r,
    export_iramuteq_corpus,
    import_directory,
    segment_uces,
)
from labiialex_pipeline.corpus import DEFAULT_UCE_SIZE
from labiialex_pipeline.matrix import DEFAULT_MIN_FREQ
from labiialex_pipeline.readers import DEFAULT_DROP_SPEAKERS

#: Diretório padrão dos dicionários do IRaMuTeQ dentro da extração.
DEFAULT_DICT_DIR = (
    Path(__file__).resolve().parents[1]
    / "extracted"
    / "internal"
    / "dictionaries"
)


def _write_uce_meta(ucis, out_dir: Path) -> Path:
    path = out_dir / "uce_meta.csv"
    var_names: list[str] = []
    for uci in ucis:
        for key in uci.variables:
            if key not in var_names:
                var_names.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["uce_id", "uci_id", *var_names, "n_tokens", "n_active"])
        for uci in ucis:
            for uce in uci.uces:
                writer.writerow(
                    [
                        uce.uce_id,
                        uci.uci_id,
                        *[uci.variables.get(v, "") for v in var_names],
                        len(uce.tokens),
                        len(uce.active_lemmas),
                    ]
                )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Preparação de corpus (método Reinert/IRaMuTeQ)")
    parser.add_argument("--corpus", required=True, help="diretório com um arquivo por entrevista")
    parser.add_argument("--out", required=True, help="diretório de saída")
    parser.add_argument("--lang", default="pt")
    parser.add_argument("--uce-size", type=int, default=DEFAULT_UCE_SIZE)
    parser.add_argument("--min-freq", type=int, default=DEFAULT_MIN_FREQ)
    parser.add_argument("--dictionaries", default=str(DEFAULT_DICT_DIR))
    parser.add_argument("--text-column", default=None,
                        help="coluna de texto em planilhas (.csv/.xlsx); auto se omitido")
    parser.add_argument("--stopwords", default=None, help="arquivo de stopwords (uma por linha)")
    parser.add_argument("--synonyms", default=None, help="CSV variante,canonico")
    parser.add_argument("--expressions", default=None, help="expressões multipalavra (uma por linha)")
    parser.add_argument("--no-adverbs", action="store_true", help="trata advérbios como suplementares")
    parser.add_argument("--unknown-supplementary", action="store_true",
                        help="trata formas fora do léxico como suplementares")
    parser.add_argument("--min-token-len", type=int, default=1, help="tamanho mínimo de token ativo")
    parser.add_argument("--anon-prefix", default="anon",
                        help="prefixo dos marcadores de anonimização a ignorar (ex.: anonmae)")
    parser.add_argument("--drop-speakers", default=None,
                        help="falantes a remover (regex separados por vírgula); "
                             "padrão: moderador,palestrante, etc.")
    parser.add_argument("--keep-all-speakers", action="store_true",
                        help="não filtra turnos por falante")
    args = parser.parse_args()

    if args.keep_all_speakers:
        drop_speakers = None
    elif args.drop_speakers:
        drop_speakers = [s.strip() for s in args.drop_speakers.split(",") if s.strip()]
    else:
        drop_speakers = list(DEFAULT_DROP_SPEAKERS)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Carregando léxico {args.lang} ...")
    lexique = Lexique.load(Path(args.dictionaries), lang=args.lang)
    print(f"      {lexique.size} formas no léxico.")
    processor = build_preprocessor(
        lexique,
        stopwords_path=args.stopwords,
        synonyms_path=args.synonyms,
        expressions_path=args.expressions,
        adverbs_active=not args.no_adverbs,
        unknown_active=not args.unknown_supplementary,
        min_token_len=args.min_token_len,
        anon_prefix=args.anon_prefix or None,
    )

    print(f"[2/4] Importando corpus de {args.corpus} ...")
    if drop_speakers:
        print(f"      filtro de turnos ativo (removendo: {', '.join(drop_speakers)})")
    ucis = import_directory(args.corpus, text_column=args.text_column,
                            drop_speakers=drop_speakers)
    print(f"      {len(ucis)} UCIs (documentos).")
    corpus_export = export_iramuteq_corpus(ucis, out_dir / "corpus_iramuteq.txt")
    print(f"      corpus limpo exportado p/ IRaMuTeQ: {corpus_export}")

    print(f"[3/4] Segmentando em UCEs (alvo {args.uce_size} tokens) ...")
    all_uces = []
    for uci in ucis:
        all_uces.extend(segment_uces(uci, processor, uce_size=args.uce_size))
    print(f"      {len(all_uces)} UCEs.")

    print(f"[4/4] Construindo matriz UCE x formas (min_freq {args.min_freq}) ...")
    dtm = build_dtm(all_uces, min_freq=args.min_freq)
    paths = export_for_r(dtm, out_dir)
    meta_path = _write_uce_meta(ucis, out_dir)
    # textos das UCEs (para os "segmentos típicos" por classe na Parte 2)
    with (out_dir / "uce_texts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["uce_id", "texto"])
        for uce in all_uces:
            writer.writerow([uce.uce_id, " ".join(uce.text.split())])

    # diagnóstico da preparação
    from labiialex_pipeline.diagnostics import (
        corpus_diagnostics, plot_uce_sizes, suggest_parameters,
    )
    diag = corpus_diagnostics(ucis, all_uces, args.min_freq)
    notes = suggest_parameters(diag)
    (out_dir / "prepared_diagnostics.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_uce_sizes(all_uces, out_dir / "uce_sizes.png")
    print("\n--- Diagnóstico da preparação ---")
    print(f"  UCEs: {diag['n_uce']} | tokens/UCE (mediana): {diag['uce_tokens']['mediana']} "
          f"| ativos: {diag['proporcao_ativos']*100:.0f}%")
    print(f"  Vocabulário ativo: {diag['vocabulario_ativo']} | hapax: {diag['hapax_pct']}% "
          f"| formas retidas (min_freq {args.min_freq}): {diag['formas_retidas_min_freq']}")
    print(f"  UCEs retidas: {diag['uces_retidas_pct']}%")
    for note in notes:
        print(f"  • {note}")

    summary = {
        "n_uci": len(ucis),
        "n_uce_total": len(all_uces),
        "n_uce_retained": dtm.shape[0],
        "n_active_forms": dtm.shape[1],
        "uce_size_target": args.uce_size,
        "min_freq": args.min_freq,
        "lang": args.lang,
        "corpus_dir": str(Path(args.corpus).resolve()),
        "dictionaries": str(Path(args.dictionaries).resolve()),
        "turn_filter": drop_speakers,
        "preprocess": {
            "stopwords": str(Path(args.stopwords).resolve()) if args.stopwords else None,
            "synonyms": str(Path(args.synonyms).resolve()) if args.synonyms else None,
            "expressions": str(Path(args.expressions).resolve()) if args.expressions else None,
            "adverbs_active": not args.no_adverbs,
            "unknown_active": not args.unknown_supplementary,
            "min_token_len": args.min_token_len,
            "anon_prefix": args.anon_prefix or None,
        },
        "outputs": {k: str(v) for k, v in {**paths, "uce_meta": meta_path}.items()},
    }
    (out_dir / "prepared.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nResumo:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
