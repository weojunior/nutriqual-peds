#!/usr/bin/env Rscript
# Emocoes via syuzhet (lexico NRC), o mesmo motor do software original.
# Uso: Rscript emotions_reference.R <docs.csv> <out.csv>
# docs.csv: ; separado, colunas 'doc_id' e 'texto'.
# Saida: doc_id + 8 emocoes (NRC) + negative/positive.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Uso: emotions_reference.R <docs.csv> <out.csv>")
docs_path <- args[[1]]; out_path <- args[[2]]

suppressWarnings(suppressMessages(library(syuzhet)))
d <- read.csv2(docs_path, header = TRUE, check.names = FALSE,
               encoding = "UTF-8", stringsAsFactors = FALSE)
texts <- as.character(d$texto)
emo <- get_nrc_sentiment(texts, language = "portuguese")
out <- cbind(doc_id = d$doc_id, emo)
write.table(out, out_path, sep = ";", row.names = FALSE,
            quote = FALSE, fileEncoding = "UTF-8")
cat(sprintf("EMOTIONS_REFERENCE_OK: %d documentos -> %s\n", nrow(out), out_path))
