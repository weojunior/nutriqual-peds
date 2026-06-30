#!/usr/bin/env Rscript
# Referencia de similitude: igraph constroi a arvore maxima da matriz de similaridade.
# Uso: Rscript simi_reference.R <sim_matrix.csv> <out_edges.csv>
# sim_matrix.csv: ; separado, 1a coluna = formas (row names), matriz forma x forma.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Uso: simi_reference.R <sim_matrix.csv> <out_edges.csv>")
sim_path <- args[[1]]; out_path <- args[[2]]

suppressWarnings(suppressMessages(library(igraph)))
d <- read.csv2(sim_path, header = TRUE, row.names = 1,
               check.names = FALSE, encoding = "UTF-8")
m <- as.matrix(d)
m[m < 0] <- 0
g <- graph_from_adjacency_matrix(m, mode = "undirected", weighted = TRUE, diag = FALSE)
# arvore maxima = MST sobre o negativo dos pesos
tree <- mst(g, weights = -E(g)$weight)
el <- as_edgelist(tree, names = TRUE)
w <- E(tree)$weight
out <- data.frame(forma_i = el[, 1], forma_j = el[, 2],
                  peso = round(w, 6), stringsAsFactors = FALSE)
write.table(out, out_path, sep = ";", row.names = FALSE,
            quote = FALSE, fileEncoding = "UTF-8")
cat(sprintf("SIMI_REFERENCE_OK: %d arestas, peso total %.4f -> %s\n",
            nrow(out), sum(w), out_path))
