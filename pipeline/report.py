#!/usr/bin/env python3
"""Gera um relatório HTML consolidado a partir da pasta de saída de um estudo.

Reúne diagnóstico, CHD (classes, palavras características, segmentos típicos),
AFC, especificidades, similitude, LDA, emoções, nuvem, n-gramas, YAKE e KWIC,
com as figuras embutidas (base64). Seções sem arquivos são omitidas.

Uso: python report.py --prepared OUT_DIR
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import pandas as pd

CSS = """
body{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:980px;margin:24px auto;
padding:0 16px;color:#1a1a1a;line-height:1.5}
h1{border-bottom:3px solid #2c7fb8;padding-bottom:6px}
h2{color:#2c7fb8;margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:4px}
h3{color:#444;margin-top:20px}
table{border-collapse:collapse;margin:10px 0;font-size:13px}
th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}
th{background:#eef5fa}
img{max-width:100%;border:1px solid #ddd;margin:8px 0}
.muted{color:#666;font-size:13px}
.warn{background:#fff3cd;border:1px solid #ffe69c;padding:8px 12px;border-radius:4px}
code{background:#f4f4f4;padding:1px 4px;border-radius:3px}
"""


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _table(path: Path, sep: str = ";", maxrows: int | None = None,
           query=None) -> str:
    if not path.exists():
        return ""
    df = pd.read_csv(path, sep=sep, encoding="utf-8")
    if query is not None:
        df = query(df)
    if maxrows:
        df = df.head(maxrows)
    return df.to_html(index=False, border=0)


def _img(path: Path, alt: str = "") -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{alt}">'


def build_report(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    parts: list[str] = ["<html><head><meta charset='utf-8'>",
                        f"<style>{CSS}</style></head><body>"]
    info = _read_json(out_dir / "prepared.json") or {}
    diag = _read_json(out_dir / "prepared_diagnostics.json") or {}

    parts.append("<h1>Relatório de análise textual</h1>")
    parts.append(f"<p class='muted'>Corpus: {info.get('n_uci','?')} documentos, "
                 f"{diag.get('n_uce','?')} UCE, {info.get('n_active_forms','?')} formas ativas. "
                 f"Língua: {info.get('lang','?')}.</p>")

    # --- diagnóstico ---
    if diag:
        parts.append("<h2>1. Diagnóstico da preparação</h2>")
        rows = {k: diag[k] for k in ("n_uci", "n_uce", "tokens_ativos", "proporcao_ativos",
                "vocabulario_ativo", "hapax_pct", "formas_retidas_min_freq", "uces_retidas_pct")
                if k in diag}
        parts.append("<table>" + "".join(
            f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows.items()) + "</table>")

    # --- CHD ---
    chd = out_dir / "chd"
    cmp = _read_json(chd / "comparison.json")
    if cmp:
        parts.append("<h2>2. Classificação Hierárquica Descendente (Reinert)</h2>")
        py = cmp.get("python", {})
        parts.append(f"<p>{py.get('n_classes','?')} classes, tamanhos "
                     f"{py.get('class_sizes',{})}.</p>")
        if cmp.get("r_reference"):
            parts.append(f"<p class='muted'>Concordância Python x R (Rand ajustado): "
                         f"{cmp['r_reference'].get('adjusted_rand_index_vs_python','?')}.</p>")
        if cmp.get("stability"):
            s = cmp["stability"]
            parts.append(f"<p class='muted'>Estabilidade: índice {s.get('stability','?')} "
                         f"(within {s.get('within_class_coassoc','?')}, "
                         f"between {s.get('between_class_coassoc','?')}).</p>")
        for w in cmp.get("warnings", []):
            parts.append(f"<p class='warn'>Aviso: {w}</p>")
        if (chd / "dendrogram.png").exists():
            parts.append("<h3>Dendrograma da CHD</h3>")
            parts.append(_img(chd / "dendrogram.png", "árvore de divisões"))
        parts.append("<h3>Palavras características</h3>")
        parts.append(_table(chd / "characteristic_terms.csv", maxrows=60))
        parts.append("<h3>Segmentos típicos</h3>")
        parts.append(_table(chd / "typical_segments.csv", maxrows=40))

    # --- AFC + especificidades ---
    afc = out_dir / "afc_spec"
    if (afc / "afc_inertia.csv").exists():
        parts.append("<h2>3. Análise Fatorial de Correspondência e especificidades</h2>")
        parts.append(_table(afc / "afc_inertia.csv"))
        parts.append(_img(afc / "afc_plane.png", "plano fatorial"))
        if (afc / "afc_coords_classes.csv").exists():
            parts.append("<h3>Classes no plano (coordenadas, COR=qualidade, CTR=contribuição)</h3>")
            parts.append(_table(afc / "afc_coords_classes.csv", maxrows=20))
        for spec in sorted(afc.glob("specificities_*.csv")):
            if spec.name.endswith("_r.csv"):
                continue
            parts.append(f"<h3>Especificidades: {spec.stem.replace('specificities_','')}</h3>")
            parts.append(_table(spec, maxrows=40))

    # --- similitude ---
    simi = out_dir / "simi"
    if (simi / "simi_nodes.csv").exists():
        parts.append("<h2>4. Análise de similitude</h2>")
        scmp = _read_json(simi / "simi_comparison.json") or {}
        parts.append(f"<p class='muted'>{scmp.get('n_communities','?')} comunidades; "
                     f"índice '{scmp.get('index','?')}'.</p>")
        parts.append(_img(simi / "simi_graph.png", "rede de similitude"))
        nodes = simi / "simi_nodes.csv"
        if nodes.exists():
            df = pd.read_csv(nodes, sep=";")
            comm = df.groupby("comunidade")["forma"].apply(
                lambda s: ", ".join(s.head(10))).reset_index()
            comm.columns = ["comunidade", "formas (top)"]
            parts.append(comm.to_html(index=False, border=0))

    # --- LDA ---
    lda = out_dir / "lda"
    if (lda / "lda_topics_py.csv").exists():
        parts.append("<h2>5. Modelagem de tópicos (LDA)</h2>")
        df = pd.read_csv(lda / "lda_topics_py.csv", sep=";")
        topics = df.groupby("topico")["forma"].apply(
            lambda s: ", ".join(s.head(10))).reset_index()
        topics.columns = ["tópico", "formas (top)"]
        parts.append(topics.to_html(index=False, border=0))
        lcmp = _read_json(lda / "lda_comparison.json") or {}
        if lcmp.get("ktuning") and (lda / "lda_ktuning.png").exists():
            kt = lcmp["ktuning"]
            parts.append(f"<p class='muted'>Escolha de k (ldatuning): sugerido "
                         f"k={kt.get('k_sugerido_combinado','?')} "
                         f"(por métrica: {kt.get('k_otimo_por_metrica',{})}).</p>")
            parts.append(_img(lda / "lda_ktuning.png", "k-tuning da LDA"))

    # --- distância de Labbé ---
    labbe = out_dir / "labbe"
    if (labbe / "labbe_heatmap.png").exists():
        parts.append("<h2>6. Distância intertextual de Labbé (entre grupos)</h2>")
        parts.append(_img(labbe / "labbe_heatmap.png", "distância de Labbé"))
        if (labbe / "labbe_clusters.png").exists():
            parts.append(_img(labbe / "labbe_clusters.png", "agrupamento dos grupos"))

    # --- emoções ---
    emo = out_dir / "emotions"
    if emo.exists():
        parts.append("<h2>7. Emoções (NRC)</h2>")
        overall = emo / "emotions_overall.png"
        if overall.exists():
            parts.append(_img(overall, "emoções médias no corpus"))
        # por variável, exceto n_participantes (só 2 níveis: 4 vs 6)
        imgs = [im for im in sorted(emo.glob("emotions_by_*.png"))
                if im.stem != "emotions_by_n_participantes"]
        for im in imgs:
            parts.append(_img(im, "emoções"))

    # --- extras ---
    extras = out_dir / "extras"
    if extras.exists():
        parts.append("<h2>8. Análises complementares</h2>")
        parts.append(_img(extras / "wordcloud.png", "nuvem de palavras"))
        parts.append(_img(extras / "heatmap_forms_classes.png", "heatmap"))
        if (extras / "bigramas.csv").exists():
            parts.append("<h3>Bigramas</h3>")
            parts.append(_table(extras / "bigramas.csv", maxrows=15))
        if (extras / "yake_keyphrases.csv").exists():
            parts.append("<h3>YAKE (termos-chave)</h3>")
            parts.append(_table(extras / "yake_keyphrases.csv", maxrows=20))

    # --- KWIC ---
    kwic = out_dir / "kwic"
    if kwic.exists():
        kfiles = sorted(kwic.glob("kwic_*.csv"))
        if kfiles:
            parts.append("<h2>9. Concordâncias (KWIC)</h2>")
            for kf in kfiles:
                parts.append(f"<h3>{kf.stem.replace('kwic_','')}</h3>")
                parts.append(_table(kf, maxrows=20))

    parts.append("<p class='muted'>Gerado pelo pipeline labiia_lex. Resultados de "
                 "análise textual automatizada; interpretar com o RESUMO_METODO.md.</p>")
    parts.append("</body></html>")
    report = out_dir / "report.html"
    report.write_text("\n".join(p for p in parts if p), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Relatório HTML consolidado")
    parser.add_argument("--prepared", required=True)
    args = parser.parse_args()
    path = build_report(Path(args.prepared))
    print(f"Relatório gerado: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
