#!/usr/bin/env Rscript
# Emocoes via lexico NRC (syuzhet), por documento.
# Uso: Rscript emotions_reference.R <docs.csv> <out.csv> [tokens|legacy]
# docs.csv: ; separado, colunas 'doc_id' e 'texto'.
# Saida: doc_id + 8 emocoes NRC + negative/positive + n_tokens + n_matched.
#
# Modo "tokens" (padrao desde a v2.1.0): casamento insensivel a acento e
# contagem de OCORRENCIAS. Motivo: syuzhet::get_nrc_sentiment separa palavras
# com "[^A-Za-z']+", o que quebra toda palavra acentuada (2609 das 3886 entradas
# do lexico portugues nunca casam), e conta cada entrada do lexico UMA vez por
# documento (tipos, nao tokens), o que torna a contagem nao aditiva.
# Modo "legacy": comportamento original do syuzhet, mantido para reproduzir os
# numeros das versoes <= 2.0.2.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Uso: emotions_reference.R <docs.csv> <out.csv> [tokens|legacy]")
docs_path <- args[[1]]; out_path <- args[[2]]
mode <- if (length(args) >= 3) args[[3]] else "tokens"

# chartr com caracteres multibyte exige locale UTF-8; o Rscript herda o locale do
# shell e pode cair em "C" (macOS/Linux sem LANG). Forcar UTF-8 aqui.
for (loc in c("en_US.UTF-8", "C.UTF-8", "pt_BR.UTF-8")) {
  ok <- suppressWarnings(Sys.setlocale("LC_CTYPE", loc))
  if (nzchar(ok)) break
}
if (!grepl("UTF-8", Sys.getlocale("LC_CTYPE"))) stop("nao foi possivel ativar um locale UTF-8")

suppressWarnings(suppressMessages(library(syuzhet)))
d <- read.csv2(docs_path, header = TRUE, check.names = FALSE,
               encoding = "UTF-8", stringsAsFactors = FALSE)
texts <- enc2utf8(as.character(d$texto))
SENTS <- c("anger", "anticipation", "disgust", "fear", "joy", "sadness",
           "surprise", "trust", "negative", "positive")

ACENTOS <- "áàâãäéèêëíìîïóòôõöúùûüçñ"
SEM_ACENTO <- "aaaaaeeeeiiiiooooouuuucn"
normalizar <- function(x) chartr(ACENTOS, SEM_ACENTO, tolower(x))

if (mode == "legacy") {
  emo <- get_nrc_sentiment(texts, language = "portuguese")
  n_tokens <- vapply(strsplit(texts, "[^A-Za-z']+"),
                     function(v) sum(nzchar(v)), integer(1))
  n_matched <- rowSums(emo[, c("anger", "anticipation", "disgust", "fear",
                               "joy", "sadness", "surprise", "trust")])
} else if (mode == "tokens") {
  lex <- syuzhet:::nrc[syuzhet:::nrc$lang == "portuguese", ]
  lex$w <- normalizar(enc2utf8(lex$word))
  lex <- unique(lex[, c("w", "sentiment")])
  por_sent <- split(lex$w, lex$sentiment)
  palavras <- enc2utf8(unique(syuzhet:::nrc$word[syuzhet:::nrc$lang == "portuguese"]))
  n_acent <- sum(grepl("[^a-z' -]", tolower(palavras)))
  cat(sprintf("lexico pt: %d palavras distintas, %d com acento (agora casaveis)\n",
              length(palavras), n_acent))
  contar <- function(tx) {
    toks <- unlist(strsplit(normalizar(tx), "[^a-z']+"))
    toks <- toks[nzchar(toks)]
    cnt <- table(toks)
    vals <- vapply(SENTS, function(s) {
      ws <- por_sent[[s]]
      if (is.null(ws)) return(0L)
      as.integer(sum(cnt[names(cnt) %in% ws]))
    }, integer(1))
    c(vals, n_tokens = length(toks))
  }
  m <- t(vapply(texts, contar, numeric(length(SENTS) + 1)))
  emo <- as.data.frame(m[, SENTS, drop = FALSE])
  n_tokens <- m[, "n_tokens"]
  n_matched <- rowSums(emo[, c("anger", "anticipation", "disgust", "fear",
                               "joy", "sadness", "surprise", "trust")])
} else {
  stop("modo desconhecido: ", mode)
}

out <- cbind(doc_id = d$doc_id, emo, n_tokens = as.integer(n_tokens),
             n_matched = as.integer(n_matched))
write.table(out, out_path, sep = ";", row.names = FALSE,
            quote = FALSE, fileEncoding = "UTF-8")
cat(sprintf("EMOTIONS_REFERENCE_OK (%s): %d documentos -> %s\n", mode, nrow(out), out_path))
