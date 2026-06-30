#!/usr/bin/env Rscript
# Distancia intertextual de Labbe (canonico IRaMuTeQ) entre colunas (textos) de
# uma tabela forma x texto. Reaproveita distance-labbe.R do Ratinaud.
# Uso: Rscript labbe_reference.R <rscripts_dir> <table.csv> <out_matrix.csv>
# table.csv: ; separado, 1a coluna = formas (row names), colunas = textos/grupos.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3)
  stop("Uso: labbe_reference.R <rscripts_dir> <table.csv> <out_matrix.csv>")
rscripts_dir <- args[[1]]; table_path <- args[[2]]; out_path <- args[[3]]

source(file.path(rscripts_dir, "distance-labbe.R"))
d <- read.csv2(table_path, header = TRUE, row.names = 1,
               check.names = FALSE, encoding = "UTF-8")
tab <- as.matrix(d)
mat <- dist.labbe(tab)              # triangular inferior; diagonal/superior = NA
mat[is.na(mat)] <- 0               # simetrizada e diagonal preenchidas em Python
write.csv2(as.data.frame(mat), out_path, row.names = TRUE, fileEncoding = "UTF-8")
cat(sprintf("LABBE_REFERENCE_OK: %d textos -> %s\n", ncol(tab), out_path))
