#!/usr/bin/env python3
"""Orquestrador: roda o pipeline inteiro sobre uma pasta de corpus e gera o relatório.

Executa, em sequência: preparação, CHD, AFC + especificidades, similitude, LDA,
emoções, complementares (nuvem, n-gramas, heatmap, YAKE), KWIC (opcional) e o
relatório HTML consolidado. Cada etapa que falhar é registrada e o fluxo continua.

Exemplo:
    python run_all.py --corpus meu_corpus --out pipeline/output/estudo8 \
        --stopwords pipeline/config_med/stopwords.txt --n-classes 4 \
        --kwic sonda,querer,suplemento
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def run_step(name: str, script: str, script_args: list[str], log_dir: Path) -> bool:
    print(f"\n=== {name} ===", flush=True)
    log = log_dir / f"{name.replace(' ', '_').lower()}.log"
    proc = subprocess.run([PY, str(HERE / script), *script_args],
                          capture_output=True, text=True, encoding="utf-8")
    log.write_text((proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or ""),
                   encoding="utf-8")
    # mostra as linhas-chave do passo
    for line in (proc.stdout or "").splitlines():
        if any(t in line for t in (">>>", "classes:", "OK", "->", "Inércia", "Estabilidade",
                                   "correlação", "Comunidades", "Tópico", "Emoções médias")):
            print("  " + line)
    if proc.returncode != 0:
        print(f"  [FALHOU] ver {log}")
        if proc.stderr.strip():
            print("  " + proc.stderr.strip().splitlines()[-1][:200])
        return False
    print(f"  [ok] log: {log.name}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Pipeline completo + relatório")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lang", default="pt")
    p.add_argument("--uce-size", type=int, default=40)
    p.add_argument("--min-freq", type=int, default=3)
    p.add_argument("--stopwords", default=None)
    p.add_argument("--synonyms", default=None)
    p.add_argument("--expressions", default=None)
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--top", type=int, default=60, help="formas na similitude")
    p.add_argument("--k", type=int, default=4, help="tópicos do LDA")
    p.add_argument("--kwic", default="", help="termos para KWIC, separados por vírgula")
    p.add_argument("--anon-prefix", default=None,
                   help="prefixo dos marcadores de anonimização a ignorar (ex.: anon)")
    p.add_argument("--drop-speakers", default=None,
                   help="falantes a remover (regex por vírgula); padrão do run_prepare se omitido")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)

    prep_args = ["--corpus", args.corpus, "--out", args.out, "--lang", args.lang,
                 "--uce-size", str(args.uce_size), "--min-freq", str(args.min_freq)]
    for flag, val in (("--stopwords", args.stopwords), ("--synonyms", args.synonyms),
                      ("--expressions", args.expressions), ("--anon-prefix", args.anon_prefix),
                      ("--drop-speakers", args.drop_speakers)):
        if val:
            prep_args += [flag, val]

    results: dict[str, bool] = {}
    results["preparacao"] = run_step("preparacao", "run_prepare.py", prep_args, logs)
    if not results["preparacao"]:
        print("\nPreparação falhou; abortando."); return 1

    results["chd"] = run_step("chd", "run_chd.py",
        ["--prepared", args.out, "--n-classes", str(args.n_classes), "--stability"], logs)
    results["afc_especificidades"] = run_step("afc_especificidades", "run_afc_spec.py",
        ["--prepared", args.out], logs)
    results["similitude"] = run_step("similitude", "run_simi.py",
        ["--prepared", args.out, "--index", "cooccurrence", "--top", str(args.top),
         "--layout", "fa2"], logs)
    results["lda"] = run_step("lda", "run_lda.py",
        ["--prepared", args.out, "--k", str(args.k), "--level", "uce",
         "--stability", "--ktuning"], logs)
    results["labbe"] = run_step("labbe", "run_labbe.py", ["--prepared", args.out], logs)
    results["emocoes"] = run_step("emocoes", "run_emotions.py", ["--prepared", args.out], logs)

    for what, extra in (("wordcloud", []), ("ngrams", ["--n", "2"]),
                        ("ngrams3", ["--n", "3"]), ("heatmap", []), ("yake", ["--n", "3"])):
        base_what = "ngrams" if what == "ngrams3" else what
        results[f"extras_{what}"] = run_step(f"extras_{what}", "run_extras.py",
            ["--prepared", args.out, "--what", base_what, *extra], logs)

    for term in [t.strip() for t in args.kwic.split(",") if t.strip()]:
        results[f"kwic_{term}"] = run_step(f"kwic_{term}", "run_kwic.py",
            ["--prepared", args.out, "--query", term], logs)

    print("\n=== relatorio ===")
    results["relatorio"] = run_step("relatorio", "report.py", ["--prepared", args.out], logs)

    ok = sum(1 for v in results.values() if v)
    print(f"\nConcluído: {ok}/{len(results)} etapas com sucesso.")
    print(f"Relatório: {out / 'report.html'}")
    falhas = [k for k, v in results.items() if not v]
    if falhas:
        print(f"Falhas (ver {logs}/): {', '.join(falhas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
