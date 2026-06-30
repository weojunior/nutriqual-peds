"""Diagnóstico da preparação do corpus, para escolher parâmetros com segurança."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .corpus import Uce, Uci


def corpus_diagnostics(
    ucis: list[Uci], uces: list[Uce], min_freq: int
) -> dict:
    """Métricas que indicam se uce_size/min_freq/n_classes são razoáveis."""
    uce_tokens = np.array([len(u.tokens) for u in uces]) if uces else np.array([0])
    uce_active = np.array([len(u.active_lemmas) for u in uces]) if uces else np.array([0])
    total_tokens = int(uce_tokens.sum())
    total_active = int(uce_active.sum())

    global_freq: Counter[str] = Counter()
    for u in uces:
        global_freq.update(u.active_lemmas)
    vocab = len(global_freq)
    hapax = sum(1 for c in global_freq.values() if c == 1)
    retained_forms = sum(1 for c in global_freq.values() if c >= min_freq)
    # UCEs que sobrevivem ao filtro (têm ao menos uma forma retida)
    retained = {f for f, c in global_freq.items() if c >= min_freq}
    nonempty = sum(1 for u in uces if retained.intersection(u.active_lemmas))

    uci_uce_counts = Counter(u.uci_id for u in uces)

    diag = {
        "n_uci": len(ucis),
        "n_uce": len(uces),
        "uce_tokens": {
            "min": int(uce_tokens.min()), "mediana": float(np.median(uce_tokens)),
            "media": round(float(uce_tokens.mean()), 1), "max": int(uce_tokens.max()),
        },
        "tokens_totais": total_tokens,
        "tokens_ativos": total_active,
        "proporcao_ativos": round(total_active / total_tokens, 3) if total_tokens else 0.0,
        "vocabulario_ativo": vocab,
        "hapax": hapax,
        "hapax_pct": round(100 * hapax / vocab, 1) if vocab else 0.0,
        "formas_retidas_min_freq": retained_forms,
        "uces_nao_vazias": nonempty,
        "uces_retidas_pct": round(100 * nonempty / len(uces), 1) if uces else 0.0,
        "uci_com_uma_uce": sum(1 for c in uci_uce_counts.values() if c <= 1),
        "top_formas": global_freq.most_common(15),
    }
    return diag


def suggest_parameters(diag: dict) -> list[str]:
    """Alertas e sugestões textuais com base no diagnóstico."""
    notes: list[str] = []
    if diag["n_uce"] < 30:
        notes.append("Poucas UCEs (<30): a CHD/AFC fica instável; junte mais textos "
                     "ou reduza --uce-size.")
    if diag["uce_tokens"]["mediana"] < 8:
        notes.append("UCEs muito curtas (mediana <8 tokens): aumente --uce-size.")
    if diag["uces_retidas_pct"] < 80:
        notes.append(f"Só {diag['uces_retidas_pct']}% das UCEs sobram após o filtro: "
                     "reduza --min-freq.")
    if diag["hapax_pct"] > 60:
        notes.append(f"{diag['hapax_pct']}% do vocabulário é hapax (freq 1): considere "
                     "--min-token-len ou revisar OCR/ruído.")
    if diag["formas_retidas_min_freq"] < 20:
        notes.append("Menos de 20 formas ativas retidas: vocabulário pequeno para "
                     "muitas classes; use poucos --n-classes.")
    if diag["uci_com_uma_uce"] > diag["n_uci"] * 0.5:
        notes.append("Mais da metade dos documentos gera só 1 UCE: textos curtos; "
                     "considere análise no nível de documento.")
    if not notes:
        notes.append("Parâmetros parecem adequados para este corpus.")
    return notes


def plot_uce_sizes(uces: list[Uce], path) -> None:
    """Histograma do tamanho das UCEs (em tokens)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [len(u.tokens) for u in uces]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sizes, bins=min(30, max(5, len(set(sizes)))), color="steelblue")
    ax.set_xlabel("tokens por UCE"); ax.set_ylabel("nº de UCEs")
    ax.set_title("Distribuição do tamanho das UCEs")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
