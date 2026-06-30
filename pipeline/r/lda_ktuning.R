#!/usr/bin/env Rscript
# Escolha do número de tópicos (k) da LDA via ldatuning::FindTopicsNumber, com as
# 4 métricas do padrão-ouro: Griffiths2004, CaoJuan2009, Arun2010, Deveaud2014.
# Uso: Rscript lda_ktuning.R <counts.csv> <k_min> <k_max> <out_csv> [seed] [method]
# counts.csv: ; separado, 1a coluna = doc_id (row names), colunas = formas (contagens).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4)
  stop("Uso: lda_ktuning.R <counts.csv> <k_min> <k_max> <out_csv> [seed] [method]")
counts_path <- args[[1]]; k_min <- as.integer(args[[2]])
k_max <- as.integer(args[[3]]); out_path <- args[[4]]
seed <- if (length(args) >= 5) as.integer(args[[5]]) else 1L
method <- if (length(args) >= 6) args[[6]] else "Gibbs"

suppressWarnings(suppressMessages({
  library(slam); library(topicmodels); library(ldatuning)
}))
d <- read.csv2(counts_path, header = TRUE, row.names = 1,
               check.names = FALSE, encoding = "UTF-8")
m <- as.matrix(d); storage.mode(m) <- "integer"
m <- m[rowSums(m) > 0, colSums(m) > 0, drop = FALSE]

dtm <- as.simple_triplet_matrix(m)
class(dtm) <- c("DocumentTermMatrix", "simple_triplet_matrix")
attr(dtm, "weighting") <- c("term frequency", "tf")

res <- FindTopicsNumber(
  dtm, topics = seq(k_min, k_max),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = method, control = list(seed = seed), verbose = FALSE)

write.table(res, out_path, sep = ";", row.names = FALSE,
            quote = FALSE, dec = ".", fileEncoding = "UTF-8")
cat(sprintf("LDA_KTUNING_OK: k=%d..%d (%s) -> %s\n", k_min, k_max, method, out_path))
