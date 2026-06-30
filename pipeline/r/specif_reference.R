#!/usr/bin/env Rscript
# Referencia de especificidades: textometry::specificities sobre formas x partes.
# Uso: Rscript specif_reference.R <table.csv> <out_long.csv>
# table.csv: ; separado, 1a coluna = formas (row names), colunas = partes.
# Saida (longo): forma;parte;score   (indice de especificidade do textometry)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Uso: specif_reference.R <table.csv> <out_long.csv>")
tab_path <- args[[1]]; out_path <- args[[2]]

suppressWarnings(suppressMessages(library(textometry)))
d <- read.csv2(tab_path, header = TRUE, row.names = 1,
               check.names = FALSE, encoding = "UTF-8")
m <- as.matrix(d)
storage.mode(m) <- "double"

sp <- specificities(m)          # matriz formas x partes (indice assinado)
sp <- as.matrix(sp)

forms <- rownames(sp); parts <- colnames(sp)
rows <- list()
idx <- 1
for (j in seq_along(parts)) {
  for (i in seq_along(forms)) {
    rows[[idx]] <- data.frame(forma = forms[i], parte = parts[j],
                              score = round(sp[i, j], 4),
                              stringsAsFactors = FALSE)
    idx <- idx + 1
  }
}
out <- do.call(rbind, rows)
write.table(out, out_path, sep = ";", row.names = FALSE,
            quote = FALSE, fileEncoding = "UTF-8")
cat(sprintf("SPECIF_REFERENCE_OK: %d formas x %d partes -> %s\n",
            length(forms), length(parts), out_path))
