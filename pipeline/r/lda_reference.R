#!/usr/bin/env Rscript
# Referencia LDA: topicmodels::LDA (VEM) sobre a matriz de contagens docs x formas.
# Uso: Rscript lda_reference.R <counts.csv> <k> <out_terms.csv> [seed] [n_terms]
# counts.csv: ; separado, 1a coluna = doc_id (row names), colunas = formas (contagens).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("Uso: lda_reference.R <counts.csv> <k> <out_terms.csv> [seed] [n_terms]")
counts_path <- args[[1]]; k <- as.integer(args[[2]]); out_path <- args[[3]]
seed <- if (length(args) >= 4) as.integer(args[[4]]) else 1L
n_terms <- if (length(args) >= 5) as.integer(args[[5]]) else 10L

suppressWarnings(suppressMessages({library(slam); library(topicmodels)}))
d <- read.csv2(counts_path, header = TRUE, row.names = 1,
               check.names = FALSE, encoding = "UTF-8")
m <- as.matrix(d); storage.mode(m) <- "integer"
m <- m[rowSums(m) > 0, colSums(m) > 0, drop = FALSE]

dtm <- as.simple_triplet_matrix(m)
class(dtm) <- c("DocumentTermMatrix", "simple_triplet_matrix")
attr(dtm, "weighting") <- c("term frequency", "tf")

res <- LDA(dtm, k = k, control = list(seed = seed))
top <- terms(res, n_terms)        # matriz n_terms x k
out <- data.frame(topico = integer(0), rank = integer(0), forma = character(0))
for (t in seq_len(ncol(top))) {
  out <- rbind(out, data.frame(topico = t, rank = seq_len(nrow(top)),
                               forma = top[, t], stringsAsFactors = FALSE))
}
write.table(out, out_path, sep = ";", row.names = FALSE,
            quote = FALSE, fileEncoding = "UTF-8")
cat(sprintf("LDA_REFERENCE_OK: k=%d, %d docs -> %s\n", k, nrow(m), out_path))
