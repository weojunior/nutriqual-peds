#!/usr/bin/env Rscript
# Referencia AFC: pacote 'ca' sobre a tabela formas x classes.
# Uso: Rscript afc_reference.R <table.csv> <out_eig.csv>
# table.csv: ; separado, primeira coluna = formas (row names), colunas = classes.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Uso: afc_reference.R <table.csv> <out_eig.csv>")
tab_path <- args[[1]]; out_path <- args[[2]]

suppressWarnings(suppressMessages(library(ca)))
d <- read.csv2(tab_path, header = TRUE, row.names = 1,
               check.names = FALSE, encoding = "UTF-8")
m <- as.matrix(d)
# remove linhas/colunas totalmente nulas (ca nao aceita)
m <- m[rowSums(m) > 0, colSums(m) > 0, drop = FALSE]

res <- ca(m)
eig <- res$sv ^ 2
pct <- 100 * eig / sum(eig)
out <- data.frame(eixo = seq_along(eig),
                  autovalor = round(eig, 8),
                  inercia_pct = round(pct, 4))
write.table(out, out_path, sep = ";", row.names = FALSE,
            quote = FALSE, fileEncoding = "UTF-8")
cat(sprintf("AFC_REFERENCE_OK: %d eixos, inercia total %.6f -> %s\n",
            length(eig), sum(eig), out_path))
