# Computational environment

## Python
- Python 3.12
- Packages: see `requirements.txt` (pandas, numpy, scipy, scikit-learn, matplotlib,
  python-docx, lxml, openpyxl, wordcloud, yake, jupyterlab, ipykernel).

## R
- R 4.4.3
- Packages: ca, ade4, cluster, proxy, Matrix, MASS, irlba, igraph, slam, stringi,
  textometry, topicmodels, ldatuning, syuzhet.

Install in R:

```r
install.packages(c("ca","ade4","cluster","proxy","Matrix","MASS","irlba","igraph",
  "slam","stringi","textometry","topicmodels","syuzhet"))
# ldatuning is archived on CRAN; install from the archive:
install.packages("https://cran.r-project.org/src/contrib/Archive/ldatuning/ldatuning_1.0.2.tar.gz",
  repos = NULL, type = "source")
```

## Third-party dependencies
Run `setup_dependencies.sh` to fetch the IRaMuTeQ dictionaries and canonical R scripts
into `extracted/internal/` (see `NOTICE`).
