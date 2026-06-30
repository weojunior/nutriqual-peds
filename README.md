# NutriQual-Peds: a reproducible pipeline for computer-assisted lexical analysis of qualitative data in pediatric nutritional therapy

[![DOI](https://zenodo.org/badge/1285083748.svg)](https://zenodo.org/badge/latestdoi/1285083748)

Reproducible pipeline for the lexical analysis of qualitative data in pediatric
nutritional therapy. It reimplements the Reinert descending hierarchical classification
(the IRaMuTeQ method) in Python and cross-validates it against the canonical R
implementation. It was used to reanalyze focus-group transcripts on the nutritional
challenges of children and adolescents undergoing chemotherapy (oral feeding, oral
nutritional supplementation, and nasoenteral tube feeding).

This repository accompanies two companion manuscripts (a clinical paper and a
methodological paper) and is provided for transparency and reproducibility.

## Relationship to labiia_lex

This is an **independent reimplementation**, not a copy of the labiia_lex software by
Rafael Cardoso Sampaio (https://github.com/cardososampaio/labiia_lex, GNU GPL v3). The
pipeline depends on the IRaMuTeQ dictionaries and the canonical R scripts distributed
with labiia_lex; those third-party components are **not redistributed here** and are
obtained separately with `setup_dependencies.sh`. See `NOTICE` for attribution.

## What is included

- `pipeline/labiialex_pipeline/` — Python package (corpus, preprocessing, CHD, AFC,
  similarity, dendrogram, etc.).
- `pipeline/run_*.py`, `pipeline/report.py` — command-line steps and orchestration.
- `pipeline/r/` — thin R interface scripts that drive the canonical IRaMuTeQ scripts.
- `pipeline/config_med/` — domain configuration (stopword list, synonym set).
- `pipeline/notebooks/` — Jupyter notebook that reproduces tables and figures
  (outputs cleared).
- `pipeline/output/estudo7/` — non-identifying outputs (document-term matrix, summary
  tables, figures).

## What is not included

The transcripts contain sensitive data (caregivers of children in cancer treatment) and
are **not shared**. Verbatim text (raw transcripts, typical segments, concordance lines,
per-document text) is excluded. The non-identifying document-term matrix and aggregate
outputs allow partial reproduction; full reproduction from raw text requires access to
the protected dataset under the original ethics approval.

## Setup

Requirements: Python 3.12 and R 4.4.3 (see `environment.md` for packages).

```bash
pip install -r requirements.txt
bash setup_dependencies.sh        # fetches the IRaMuTeQ dictionaries and R scripts
```

## Usage

```bash
cd pipeline
python run_prepare.py --corpus <CORPUS_DIR> --out output/estudo7 \
  --lang pt --uce-size 40 --min-freq 3 \
  --stopwords config_med/stopwords.txt --synonyms config_med/synonyms.csv \
  --anon-prefix anon
python run_all.py --corpus <CORPUS_DIR> --out output/estudo7 --n-classes 3 ...
python run_saturation.py --prepared output/estudo7 --n-classes 3
```

The notebook `pipeline/notebooks/01_reproducao_analise_lexical.ipynb` runs the whole
flow and renders the results.

## License

GNU General Public License v3.0 or later (see `LICENSE`), compatible with the reference
software.

## Author

Wilson E. Oliveira Junior, MD, PhD.
[ORCID 0000-0001-9812-6282](https://orcid.org/0000-0001-9812-6282) ·
[Scopus 57225293188](https://www.scopus.com/authid/detail.uri?authorId=57225293188) ·
[Google Scholar](https://scholar.google.com/citations?user=2WQS9QIAAAAJ&hl=pt-BR)

## How to cite

Archived on Zenodo, DOI [10.5281/zenodo.21070825](https://doi.org/10.5281/zenodo.21070825)
(see `CITATION.cff`). The badge above always resolves to the latest version.

## Acknowledgements

We thank the developers of the labiia_lex software, which provided the reference
dictionaries, the canonical R scripts, and the conceptual basis for this
reimplementation: Rafael Cardoso Sampaio, together with Anderson Henrique (USP), Dalson
Figueiredo (UFPE), Ian Batista (Carter Center), Leonardo Nascimento (LabUFBA), and
Nilton Sainz (UFPR), and the collaborators of labiia_lab. We acknowledge the IRaMuTeQ
method by Pierre Ratinaud.

## Ethics

The original study was approved by the institutional research ethics committee (CAAE
81284724.9.0000.5437). No individually identifying data are included in this repository.
