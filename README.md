# NutriQual-Peds: a reproducible pipeline for computer-assisted lexical analysis of qualitative data in pediatric nutritional therapy

[![DOI](https://zenodo.org/badge/1410943979.svg)](https://zenodo.org/badge/latestdoi/1410943979)

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

- `pipeline/labiialex_pipeline/`: Python package (corpus, preprocessing, CHD, AFC,
  similarity, dendrogram, etc.).
- `pipeline/run_*.py`, `pipeline/report.py`: command-line steps and orchestration.
- `pipeline/r/`: thin R interface scripts that drive the canonical IRaMuTeQ scripts.
- `pipeline/config_med/`: domain configuration (stopword list, synonym set).
- `pipeline/notebooks/`: Jupyter notebook that reproduces tables and figures
  (outputs cleared).
- `pipeline/output/estudo7/`: non-identifying outputs (document-term matrix, summary
  tables, figures). Unit metadata is limited to group labels and token counts.
  Emotion results by caregiver are published only as a summary by emotion
  (`emotions_speakers/emotions_speakers_summary.csv`).

## What is not included

The transcripts contain sensitive data (caregivers of children in cancer treatment) and
are **not shared**. Verbatim text (raw transcripts, the segmented corpus, typical
segments, concordance lines, per-document text) is excluded. So are the per-group
demographic and clinical metadata used in the specificity analyses and the
per-caregiver emotion counts, which are available under controlled access with the
corresponding author. The non-identifying document-term matrix and aggregate outputs
allow partial reproduction. Full reproduction from raw text requires access to the
protected dataset under the original ethics approval.

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
python run_calibracao_saturacao.py --prepared output/estudo7 --n-classes 3
python run_emotions.py --prepared output/estudo7            # NRC emotions, corrected matching
python run_emotions_por_falante.py --transcripts <TRANSCRIPTS_DIR> \
  --out output/estudo7/emotions_speakers                    # publish only the summary file
```

`run_calibracao_saturacao.py` calibrates the saturation indicators against the
instrument itself: resampling stability of the classification (100 resamples), a
null baseline for the incremental adjusted Rand index (nested random subsets with
the same sizes as the chronological steps, with the percentile of each observed
value), and a Heaps model fitted to the vocabulary growth.

The notebook `pipeline/notebooks/01_reproducao_analise_lexical.ipynb` runs the whole
flow and renders the results.

## Changelog

### 2.1.1 (2026-10-08)

Removal of data that should not have been public. No analysis result changed.

- A first name of a person mentioned in the transcripts was removed from
  `pipeline/config_med/stopwords.txt`. The name does not occur in the anonymized
  corpus, so `dtm.csv` and `forms.csv` are byte-identical to version 2.1.0 (SHA-256
  `b7fcb28ea7c3dc8abb01c53c452db3edb4b2798b1af3a51f2f2fbad0f6002fdd` and
  `55d285e17d3f9ad0e93e8234772ea05890f31dcc3a7ef7b29ace96faebaaa0ec`).
- Per-group demographic and clinical metadata were removed from the outputs: the
  metadata columns of `chd/chd_classes.csv`, the specificity and emotion tables by
  metadata variable, and the matching entries of `afc_spec/afc_spec_comparison.json`.
  Results by group (`grupo`) and the class of each segment are unchanged.
- The per-caregiver emotion file was removed. The summary by emotion is kept.
- `CITATION.cff` pointed to the DOI of version 1.0.0. It now carries no DOI (see How
  to cite).
- `.gitignore` now blocks outputs by metadata variable and per-caregiver files.

The repository history was rewritten without these files, and the removal of the
earlier Zenodo versions was requested.

### 2.1.0 (2026-09-07)

Correction of the NRC emotion analysis. `syuzhet::get_nrc_sentiment()` splits words on
`[^A-Za-z']+`, which breaks every accented word, so 1,613 of the 5,501 distinct entries
of the Portuguese NRC lexicon could never be matched, and it counts each lexicon entry
once per document (distinct words, not occurrences), which makes the counts
non-additive across documents. `pipeline/r/emotions_reference.R` now matches words
after removing diacritics and counts occurrences (mode `tokens`, the default). The
previous behaviour is kept as mode `legacy` (`run_emotions.py --mode legacy`) and its
per-document output is preserved as `emotions/emotions_per_doc_legacy_v2.0.2.csv`. The
corrected matching was cross-checked against an independent Python implementation
(Unicode NFKD normalisation): identical counts for all eight emotions. The script also
forces a UTF-8 locale, because `Rscript` inherits the shell locale and a `C` locale
corrupts multibyte characters. New outputs: `emotions/emotions_summary.csv` (totals,
pooled share and mean per-document share) and `emotions_speakers/` (per-caregiver
aggregates produced by `run_emotions_por_falante.py`; the per-speaker text is never
written to disk). Ranking of the three leading emotions (sadness, fear, trust) is
unchanged; joy and anticipation move ahead of disgust and anger.

### 2.0.2 (2026-08-26)

Sanitised per-group metadata and hardened the exclusion of sensitive data.

### 1.0.0

Initial release.

## License

GNU General Public License v3.0 or later (see `LICENSE`), compatible with the reference
software.

## Author

Wilson E. Oliveira Junior, MD, PhD.
[ORCID 0000-0001-9812-6282](https://orcid.org/0000-0001-9812-6282) ·
[Scopus 57225293188](https://www.scopus.com/authid/detail.uri?authorId=57225293188) ·
[Google Scholar](https://scholar.google.com/citations?user=2WQS9QIAAAAJ&hl=pt-BR)

## How to cite

Archived on Zenodo. The DOI badge at the top of this page resolves to the latest
version. Cite the DOI of the version you used, shown on its Zenodo page.

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
