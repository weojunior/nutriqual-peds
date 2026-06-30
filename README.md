# Reproducible lexical analysis pipeline (Reinert/IRaMuTeQ, Python and R)

Reproducible pipeline that reimplements the Reinert descending hierarchical
classification (the IRaMuTeQ method) in Python and cross-validates it against the
canonical R implementation. It was used to reanalyze focus-group transcripts on the
nutritional challenges of children and adolescents undergoing chemotherapy.

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

## How to cite

See `CITATION.cff`. The archived version has a DOI (Zenodo); cite that DOI.

## Acknowledgements

We thank Professor Rafael Cardoso Sampaio for making the labiia_lex repository publicly
available, and we acknowledge the IRaMuTeQ method by Pierre Ratinaud.

## Ethics

The original study was approved by the institutional research ethics committee (CAAE
81284724.9.0000.5437). No individually identifying data are included in this repository.
