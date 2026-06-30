#!/usr/bin/env Rscript
# Referencia canonica: roda o CHD.R original do IRaMuTeQ sobre dtm.csv.
# Uso: Rscript chd_reference.R <rscripts_dir> <dtm.csv> <n_classes> <out_csv> [log]
#
# Saida: CSV uce_id;classe (particao final do CHD canonico).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Uso: chd_reference.R <rscripts_dir> <dtm.csv> <n_classes> <out_csv> [log]")
}
rscripts_dir <- args[[1]]
dtm_path     <- args[[2]]
n_classes    <- as.integer(args[[3]])
out_csv      <- args[[4]]
log_path     <- if (length(args) >= 5) args[[5]] else tempfile(fileext = ".log")

# O CHD.R original e' verboso (muitos print); redireciona o ruido para um log.
log_con <- file(log_path, open = "wt")
sink(log_con, type = "output")
sink(log_con, type = "message")

ok <- TRUE
err_msg <- ""
result_classes <- NULL
uce_ids <- NULL

try_run <- function() {
  src <- function(f) source(file.path(rscripts_dir, f))
  # Dependencias do CHD canonico (boostana = AFC; svd.method 'svdR' = base R).
  if (file.exists(file.path(rscripts_dir, "Rfunct.R"))) src("Rfunct.R")
  src("anacor.R")
  src("CHD.R")

  d <- read.csv2(dtm_path, header = TRUE, row.names = 1,
                 check.names = FALSE, encoding = "UTF-8")
  m <- as.matrix(d)
  storage.mode(m) <- "integer"
  # O CHD.R original exige IDs de UCE INTEIROS (faz as.integer(rownames)).
  # Usamos 1..n como rownames e guardamos os IDs originais para mapear de volta.
  orig_ids <- rownames(m)
  rownames(m) <- as.character(seq_along(orig_ids))

  # x iteracoes de bipartição da maior classe => x+1 classes.
  x <- max(1L, n_classes - 1L)
  res <- CHD(m, x = x, svd.method = "svdR")
  dout <- res$n1
  if (is.null(dim(dout))) {
    part <- as.integer(dout)
    surv <- if (!is.null(names(dout))) as.integer(names(dout)) else seq_along(part)
  } else {
    part <- as.integer(dout[, ncol(dout)])
    surv <- if (!is.null(rownames(dout))) as.integer(rownames(dout)) else seq_along(part)
  }
  keep <- !is.na(surv) & !is.na(part)
  result_classes <<- part[keep]
  uce_ids <<- orig_ids[surv[keep]]
}

res_try <- tryCatch(try_run(), error = function(e) {
  ok <<- FALSE
  err_msg <<- conditionMessage(e)
})

sink(type = "message")
sink(type = "output")
close(log_con)

if (!ok || is.null(result_classes)) {
  cat(sprintf("CHD_REFERENCE_FAILED: %s\n", err_msg))
  cat(sprintf("(ver log: %s)\n", log_path))
  quit(status = 3)
}

# Renumera a particao para 1..K por tamanho decrescente (rotulo arbitrario).
tab <- sort(table(result_classes), decreasing = TRUE)
relabel <- setNames(seq_along(tab), names(tab))
final <- relabel[as.character(result_classes)]

out <- data.frame(uce_id = uce_ids, classe = as.integer(final),
                  stringsAsFactors = FALSE)
write.table(out, out_csv, sep = ";", row.names = FALSE,
            quote = FALSE, fileEncoding = "UTF-8")
cat(sprintf("CHD_REFERENCE_OK: %d UCEs, %d classes -> %s\n",
            nrow(out), length(unique(final)), out_csv))
