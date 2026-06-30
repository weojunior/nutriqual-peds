#!/usr/bin/env python3
"""Gera o notebook de reprodução da análise lexical (estudo8).

Monta um .ipynb com nbformat: uma seção por etapa do pipeline, cada uma com
explicação em markdown, a chamada do script validado e a exibição das tabelas e
figuras geradas. Reexecutar o notebook reproduz todos os resultados (Python e R).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT_NB = HERE / "01_reproducao_analise_lexical.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ---------------------------------------------------------------- cabeçalho
md(r'''
# Reprodução da análise lexical dos grupos focais

Reanálise dos **7 grupos focais** com pais/cuidadores de pacientes oncológicos
pediátricos (desafios alimentares: via oral, suplementação/TNO e sonda/SNE),
pelo método de Reinert/IRaMuTeQ, com **validação cruzada Python × R**.

> Nota: o corpus tinha 8 arquivos, mas a distância de Labbé detectou que
> `grupo4.docx` era cópia do `grupo2.docx` (distância 0,015 vs 0,42-0,56 nos demais
> pares). São 7 grupos reais (G1, G2, G3, G5, G6, G7, G8); a duplicata foi removida.

Este notebook reexecuta, passo a passo, todo o pipeline e exibe as tabelas e
figuras. Rodar **Célula ▸ Executar tudo** regenera os resultados em
`pipeline/output/estudo8/`.

**Requisitos**
- Kernel **labiia_lex (Python 3.12)** (o venv do projeto, em `~/Documents/labiia_lex/.venv`).
- **R 4.4** com os pacotes: `ca`, `ade4`, `cluster`, `proxy`, `Matrix`, `MASS`,
  `irlba`, `igraph`, `slam`, `stringi`, `textometry`, `topicmodels`, `syuzhet`.

**Dados sensíveis.** As transcrições contêm relatos de pais de crianças em
tratamento oncológico. Os nomes foram anonimizados manualmente com o prefixo
`anon` (tratados como suplementares). O material e os achados são **inéditos**;
não divulgar antes da publicação.
''')

# ---------------------------------------------------------------- 0. config
md(r'''
## 0. Configuração

Todos os caminhos e parâmetros num só lugar. Para reanalisar outro corpus, basta
trocar `CORPUS`/`OUT` e reexecutar.
''')

code(r'''
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

pd.set_option("display.max_colwidth", 90)
pd.set_option("display.max_rows", 60)

# --- caminhos ---
BASE = Path.home() / "Documents" / "labiia_lex"
PIPELINE = BASE / "pipeline"
CORPUS = BASE / "meu_corpus"
CONFIG = PIPELINE / "config_med"
OUT = PIPELINE / "output" / "estudo7"   # corpus de 7 grupos (duplicata G2/G4 removida)
assert PIPELINE.is_dir(), f"Pipeline não encontrado em {PIPELINE}"

# --- parâmetros da análise ---
LANG = "pt"
UCE_SIZE = 40            # tokens-alvo por UCE
MIN_FREQ = 3             # frequência mínima de uma forma ativa
N_CLASSES = 3            # classes da CHD (adotado: 2 só descola outlier; 4 acrescenta outro)
LDA_K = 4               # tópicos da LDA
SIMI_TOP = 60           # formas na análise de similitude
THEMATIC_K = 3          # grupo da saturação temática declarada (análise manual original)
KWIC_TERMS = ["sonda", "suplemento", "querer", "açúcar", "medo"]

STOPWORDS = CONFIG / "stopwords.txt"
SYNONYMS = CONFIG / "synonyms.csv"
ANON_PREFIX = "anon"

print("Kernel Python:", sys.executable)
print("Saída:", OUT)
''')

code(r'''
# Helpers: executar um script do pipeline e exibir tabelas/figuras.
_MARKERS = (">>>", " OK", "OK:", "OK_", "->", "classes:", "Classe ", "Inércia",
            "Estabil", "Concord", "Comunidades", "Tópico", "Saturação", "[ok]",
            "[FALHOU]", "Nuvem", "Heatmap", "YAKE", "Bigramas", "Trigramas",
            "Emoções", "Árvore", "criterio", "ESTRUTURAL", "TEMÁTICA", "vocab")


def run(script: str, *script_args) -> subprocess.CompletedProcess:
    """Roda PIPELINE/script com o Python do kernel; mostra as linhas-chave."""
    cmd = [sys.executable, str(PIPELINE / script), *map(str, script_args)]
    print("$", script, " ".join(str(a) for a in script_args))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in (proc.stdout or "").splitlines():
        if any(m in line for m in _MARKERS):
            print("  " + line)
    if proc.returncode != 0:
        print("  [ERRO] returncode", proc.returncode)
        print(textwrap.indent((proc.stderr or "").strip()[-1500:], "    "))
    return proc


def show_csv(relpath: str, sep: str = ";", n: int = 10, cols=None) -> pd.DataFrame:
    df = pd.read_csv(OUT / relpath, sep=sep)
    view = df[cols] if cols else df
    display(view.head(n) if n else view)
    return df


def show_img(relpath: str, width: int = 760) -> None:
    display(Image(filename=str(OUT / relpath), width=width))


def load_json(relpath: str) -> dict:
    return json.loads((OUT / relpath).read_text(encoding="utf-8"))
''')

# ---------------------------------------------------------------- 1. prepare
md(r'''
## 1. Preparação do corpus

Importa os 8 arquivos `.docx`, remove os turnos do moderador, limpa marcações de
transcrição (inaudível, risos, rótulos de participante), segmenta cada entrevista
em **UCE** (unidades de contexto elementar de ~40 tokens, respeitando sentenças)
e extrai as **formas ativas** aplicando o léxico do IRaMuTeQ mais a configuração
de domínio (stopwords, sinônimos resolvidos transitivamente, marcadores `anon`).
A matriz documento-termo (DTM) retém formas com frequência ≥ 3.
''')

code(r'''
run("run_prepare.py", "--corpus", CORPUS, "--out", OUT, "--lang", LANG,
    "--uce-size", UCE_SIZE, "--min-freq", MIN_FREQ,
    "--stopwords", STOPWORDS, "--synonyms", SYNONYMS, "--anon-prefix", ANON_PREFIX)
''')

code(r'''
prep = load_json("prepared.json")
print(f"UCIs (entrevistas): {prep['n_uci']}")
print(f"UCEs retidas:       {prep['n_uce_retained']} (de {prep['n_uce_total']})")
print(f"Formas ativas:      {prep['n_active_forms']}")

meta = pd.read_csv(OUT / "uce_meta.csv", sep=";")
display(meta["grupo"].value_counts().sort_index().rename("UCEs").to_frame().T)
''')

# ---------------------------------------------------------------- 2. CHD
md(r'''
## 2. CHD — Classificação Hierárquica Descendente (Reinert)

Núcleo do método: particiona as UCEs em classes lexicais por divisões sucessivas
sobre o primeiro fator da AFC, escolhendo o corte de máxima inércia (qui-quadrado)
e realocando segmentos. Cada classe é descrita pelas **formas características**
(maior qui-quadrado de associação). A partição em Python é confrontada com o
`CHD.R` canônico do IRaMuTeQ (índice de Rand ajustado) e a estabilidade é estimada
por subamostragem.

**Número de classes.** Adotamos **3**. Com 2, a primeira divisão apenas descola um
fragmento residual de 6 UCEs (fala regional) e deixa o resto num bloco único; com
4, acrescenta-se um segundo fragmento de 6 UCEs (cateter/infecção). Com 3 emergem
as duas classes substantivas (oral × enteral) que a similitude e a LDA também
recuperam.
''')

code(r'''
run("run_chd.py", "--prepared", OUT, "--n-classes", N_CLASSES, "--stability")
''')

code(r'''
cls = pd.read_csv(OUT / "chd/chd_classes.csv", sep=";")
display(cls["classe_py"].value_counts().sort_index().rename("UCEs por classe").to_frame().T)

ct = pd.read_csv(OUT / "chd/characteristic_terms.csv", sep=";")
pos = ct[ct["sinal"] == "+"].sort_values(["classe", "qui2"], ascending=[True, False])
print("Formas características (qui² +, top 10):")
for c, g in pos.groupby("classe"):
    print(f"  Classe {c}: " + ", ".join(g["forma"].head(10)))

comp = load_json("chd/comparison.json")
print(f"\nConcordância Python × R (Rand ajustado): {comp['r_reference']['adjusted_rand_index_vs_python']}")
s = comp["stability"]
print(f"Estabilidade: {s['stability']} (coesão interna {s['within_class_coassoc']} / entre classes {s['between_class_coassoc']})")
''')

code(r'''
# Segmentos típicos (UCEs mais representativas de cada classe)
ts = pd.read_csv(OUT / "chd/typical_segments.csv", sep=";")
show = ts[ts["rank"] <= 2][["classe", "score", "texto"]]
display(show)
''')

md(r'''
**Dendrograma da CHD** (árvore de divisões descendentes): a raiz no topo, as classes
terminais embaixo, e o qui-quadrado de cada corte. Mostra em que ordem o corpus foi
subdividido.
''')

code(r'''
show_img("chd/dendrogram.png", width=700)
''')

# ---------------------------------------------------------------- 3. AFC
md(r'''
## 3. AFC e especificidades

A análise fatorial de correspondências (AFC) projeta classes e formas num plano
fatorial. As **especificidades** (cálculo de Lafon, hipergeométrico) indicam quais
formas estão sobre/sub-representadas em cada modalidade das variáveis (grupo,
sexo, escolaridade, região, tempo de tratamento, desnutrição, SNE). A inércia por
eixo é confrontada com o pacote `ca` do R.
''')

code(r'''
run("run_afc_spec.py", "--prepared", OUT)
''')

code(r'''
afc = pd.read_csv(OUT / "afc_spec/afc_inertia.csv", sep=";")
display(afc)
show_img("afc_spec/afc_plane.png")
# COR (qualidade, cos^2) e CTR (contribuição) de cada classe nos eixos
print("Classes no plano fatorial (COR = qualidade, CTR = contribuição):")
display(pd.read_csv(OUT / "afc_spec/afc_coords_classes.csv", sep=";"))
''')

code(r'''
# Formas mais específicas de cada grupo (especificidade positiva)
spg = pd.read_csv(OUT / "afc_spec/specificities_grupo.csv", sep=";")
top = (spg[spg["sign"] == "+"].sort_values(["part", "score"], ascending=[True, False])
       .groupby("part").head(6))
for part, g in top.groupby("part"):
    print(f"{part}: " + ", ".join(g["form"]))
''')

# ---------------------------------------------------------------- 4. similitude
md(r'''
## 4. Análise de similitude

Constrói o grafo de coocorrência das formas mais frequentes, extrai a **árvore de
peso máximo** (estrutura de associações) e detecta **comunidades** (Louvain). A
árvore máxima é confrontada com o R (`igraph`) pela sobreposição de arestas
(Jaccard).
''')

code(r'''
run("run_simi.py", "--prepared", OUT, "--index", "cooccurrence",
    "--top", SIMI_TOP, "--layout", "fa2")
''')

code(r'''
nodes = pd.read_csv(OUT / "simi/simi_nodes.csv", sep=";")
print("Comunidades (formas por ordem de frequência):")
for c, g in nodes.groupby("comunidade"):
    formas = g.sort_values("frequencia", ascending=False)["forma"].head(8)
    print(f"  Comunidade {c}: " + ", ".join(formas))

sc = load_json("simi/simi_comparison.json")
print(f"\nÁrvore máxima Python × R (Jaccard das arestas): {sc['mst_edge_jaccard_py_vs_r']}"
      f" | nº de comunidades: {sc['n_communities']}")
show_img("simi/simi_graph.png")
''')

# ---------------------------------------------------------------- 5. LDA
md(r'''
## 5. LDA (modelagem de tópicos, complementar)

Alocação latente de Dirichlet sobre as UCEs, como leitura complementar à CHD. Por
ser estocástica, a LDA não é reproduzível como a CHD/AFC: reportamos a
sobreposição de termos Python × R e a estabilidade entre sementes apenas como
contexto. O motor canônico das classes continua sendo a CHD.
''')

code(r'''
run("run_lda.py", "--prepared", OUT, "--k", LDA_K, "--level", "uce",
    "--stability", "--ktuning", "--k-min", 2, "--k-max", 10)
''')

code(r'''
tp = pd.read_csv(OUT / "lda/lda_topics_py.csv", sep=";")
for t, g in tp.sort_values(["topico", "rank"]).groupby("topico"):
    print(f"Tópico {t}: " + ", ".join(g.sort_values("rank")["forma"].head(10)))

lc = load_json("lda/lda_comparison.json")
print(f"\nSobreposição média Python × R: {lc['mean_topic_top_terms_jaccard']}"
      f" | estabilidade entre sementes: {lc['topic_stability_jaccard']}")
if lc.get("ktuning"):
    print("k-tuning (ldatuning):", json.dumps(lc["ktuning"], ensure_ascii=False))
''')

md(r'''
**Escolha de k (ldatuning)**: as 4 métricas de seleção do número de tópicos
(Griffiths2004, CaoJuan2009, Arun2010, Deveaud2014). Quando elas divergem muito, a
estrutura de tópicos é mal-determinada, o que reforça tratar a LDA como complementar.
''')

code(r'''
show_img("lda/lda_ktuning.png", width=720)
''')

# ---------------------------------------------------------------- 6. Labbé
md(r'''
## 6. Distância intertextual de Labbé (entre grupos)

Distância de Labbé (canônica IRaMuTeQ) entre os 7 grupos, a partir da tabela
forma × grupo. Mede quão diferente é o vocabulário de cada grupo em relação aos
demais, normalizando pelo tamanho. Foi essa análise que detectou a duplicata
G2/G4. O heatmap mostra as distâncias par a par e o agrupamento reúne os grupos
lexicalmente próximos.
''')

code(r'''
run("run_labbe.py", "--prepared", OUT, "--by", "grupo")
show_img("labbe/labbe_heatmap.png", width=620)
show_img("labbe/labbe_clusters.png", width=720)
''')

# ---------------------------------------------------------------- 7. emoções
md(r'''
## 7. Emoções e sentimento (complementar)

Léxico NRC (via `syuzhet`, R) aplicado a cada documento. Reportamos a **média
geral das emoções no corpus inteiro** (8 categorias de Plutchik). O sentimento em
português é frágil, então estes valores entram apenas como **pista**, não como
medida principal.
''')

code(r'''
run("run_emotions.py", "--prepared", OUT)
em = pd.read_csv(OUT / "emotions/emotions_per_doc.csv", sep=";")
emo_cols = ["raiva", "antecipação", "nojo", "medo", "alegria", "tristeza",
            "surpresa", "confiança", "negative", "positive"]
display(em[emo_cols].mean().round(3).rename("média no corpus").to_frame().T)
# gráfico único: média geral das 8 emoções no corpus (sem estratificar)
show_img("emotions/emotions_overall.png")
''')

# ---------------------------------------------------------------- 7. extras
md(r'''
## 8. Complementares: nuvem, n-gramas, heatmap, YAKE

Nuvem de palavras das formas ativas, bigramas/trigramas mais frequentes, heatmap
formas × classes da CHD e extração de expressões-chave (YAKE).
''')

code(r'''
for what, extra in [("wordcloud", []), ("ngrams", ["--n", "2"]),
                    ("ngrams", ["--n", "3"]), ("heatmap", []), ("yake", ["--n", "3"])]:
    run("run_extras.py", "--prepared", OUT, "--what", what, *extra)
''')

code(r'''
show_img("extras/wordcloud.png")
show_img("extras/heatmap_forms_classes.png")
print("Bigramas mais frequentes:")
display(pd.read_csv(OUT / "extras/bigramas.csv", sep=";").head(12))
print("Expressões-chave (YAKE; menor score = mais relevante):")
display(pd.read_csv(OUT / "extras/yake_keyphrases.csv", sep=";").head(12))
''')

# ---------------------------------------------------------------- 8. KWIC
md(r'''
## 9. KWIC — concordâncias (palavra no contexto)

Para cada termo de interesse, lista as ocorrências com o contexto à esquerda e à
direita, preservando o grupo de origem.
''')

code(r'''
for term in KWIC_TERMS:
    run("run_kwic.py", "--prepared", OUT, "--query", term)
''')

code(r'''
for term in KWIC_TERMS:
    path = OUT / f"kwic/kwic_{term}.csv"
    if path.exists():
        df = pd.read_csv(path, sep=";")
        print(f"=== {term} ({len(df)} ocorrências) ===")
        display(df[["grupo", "esquerda", "palavra", "direita"]].head(6))
''')

# ---------------------------------------------------------------- 9. saturação
md(r'''
## 10. Saturação lexical incremental

Mede, ao acrescentar cada grupo (ordem cronológica e ordens aleatórias), quando a
estrutura lexical estabiliza, e compara com a saturação **temática** declarada na
análise manual original (grupo 3). Critérios:

- **vocabulário**: proporção de formas ativas novas (descritivo; raramente satura);
- **estrutura CHD**: Rand ajustado entre passos consecutivos (≥ 0,80);
- **formas características**: Jaccard entre passos consecutivos (≥ 0,80);
- **cobertura** das 4 categorias temáticas (alimentação, higiene, sintomas GI, terapias/SNE).
''')

code(r'''
run("run_saturation.py", "--prepared", OUT, "--n-classes", N_CLASSES,
    "--runs", 20, "--thematic-k", THEMATIC_K)
''')

code(r'''
sat = pd.read_csv(OUT / "saturation/saturation_chronological.csv", sep=";")
display(sat[["k", "grupos", "vocab", "new_ratio", "chd_ari", "char_jaccard",
             "cat_alimentacao", "cat_higiene", "cat_sintomas_gi", "cat_terapias_sne"]])
show_img("saturation/saturation_curve.png")

ss = load_json("saturation/saturation_summary.json")
print("Saturação por critério:", json.dumps(ss["saturacao_por_criterio"], ensure_ascii=False))
print("Temática declarada (manual): grupo", ss["saturacao_tematica_declarada"])
print("Estrutural em ordens aleatórias:", json.dumps(ss["saturacao_lexical_aleatoria"], ensure_ascii=False))
''')

# ---------------------------------------------------------------- 10. relatório
md(r'''
## 11. Relatório HTML consolidado

Reúne todas as etapas num único `report.html` (abrir no navegador).
''')

code(r'''
run("report.py", "--prepared", OUT)
print("Relatório:", OUT / "report.html")
# Para visualizar embutido, descomente:
# from IPython.display import IFrame
# IFrame(str(OUT / "report.html"), width="100%", height=600)
''')

# ---------------------------------------------------------------- 11. atalho
md(r'''
## 12. Atalho: reexecutar tudo de uma vez

As células acima já rodaram cada etapa. Para refazer o pipeline inteiro num único
comando (equivalente a tudo acima), troque a variável abaixo para `True`.
''')

code(r'''
REEXECUTAR_TUDO = False
if REEXECUTAR_TUDO:
    run("run_all.py", "--corpus", CORPUS, "--out", OUT,
        "--stopwords", STOPWORDS, "--synonyms", SYNONYMS, "--anon-prefix", ANON_PREFIX,
        "--n-classes", N_CLASSES, "--k", LDA_K, "--top", SIMI_TOP,
        "--kwic", ",".join(KWIC_TERMS))
''')

# ---------------------------------------------------------------- metadados
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "labiia_lex (Python 3.12)",
    "language": "python",
    "name": "labiialex",
}
nb["metadata"]["language_info"] = {"name": "python"}

OUT_NB.write_text(nbf.writes(nb), encoding="utf-8")
print("Notebook escrito em", OUT_NB, "com", len(cells), "células")
