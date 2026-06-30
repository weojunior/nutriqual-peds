#!/usr/bin/env bash
# Fetches the third-party IRaMuTeQ dictionaries and canonical R scripts required by the
# pipeline, from the labiia_lex repository (Rafael Cardoso Sampaio, GNU GPL v3). These
# components are NOT redistributed in this repository; they are placed under
# extracted/internal/ so the pipeline can run.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
echo "Cloning labiia_lex (dictionaries and R scripts only)..."
# checkout may warn about git-lfs binaries (JRE, gephi jar); those are not needed.
git clone --depth 1 https://github.com/cardososampaio/labiia_lex "$TMP" || true
mkdir -p "$HERE/extracted/internal"
cp -R "$TMP/dictionaries" "$HERE/extracted/internal/" && echo "  dictionaries: ok"
cp -R "$TMP/Rscripts"     "$HERE/extracted/internal/" && echo "  Rscripts: ok"
rm -rf "$TMP"
echo "Done. Third-party components are under extracted/internal/ (see NOTICE)."
