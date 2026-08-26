#!/usr/bin/env python3
"""Calibração dos indicadores de saturação lexical contra o próprio instrumento.

Complementa run_saturation.py com três controles:
  1. Estabilidade da CHD por reamostragem (padrão: 100 réplicas a 85% das UCEs),
     com a distribuição do índice de Rand ajustado de cada réplica contra a
     partição de referência do corpus completo.
  2. Linha de base nula do ARI incremental: pares de subconjuntos aleatórios
     aninhados com os mesmos tamanhos dos passos cronológicos, medindo o teto
     esperado do indicador sob variabilidade de reamostragem, sem estrutura de
     grupos. Reporta o percentil do valor observado em cada passo.
  3. Ajuste de Heaps (V = K * N^beta) por mínimos quadrados em escala log-log
     sobre as ocorrências ativas acumuladas, com a proporção de formas novas
     predita passo a passo.

Uso:
    python run_calibracao_saturacao.py --prepared OUT_DIR [--n-classes 3]
        [--min-freq 3] [--runs-estabilidade 100] [--replicas-nula 50] [--seed 0]

Saídas em OUT_DIR/saturation/calibracao/: calibracao_resultados.json e
linha_base_nula.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from labiialex_pipeline.reinert import run_chd  # noqa: E402

SUBAMOSTRA_ESTABILIDADE = 0.85
MIN_UCES_PARA_CHD = 10
MIN_UCES_COMUNS = 5


def carregar_dados(prepared: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carrega a matriz documento-termo e o rótulo de grupo de cada UCE.

    Args:
        prepared: diretório de saída de run_prepare.py, com dtm.csv e
            uce_meta.csv.

    Returns:
        Tupla (matriz, formas, grupos), com a matriz de contagens por UCE, o
        vetor de formas e o vetor do grupo de cada UCE.
    """
    frame = pd.read_csv(prepared / "dtm.csv", sep=";", dtype=str, encoding="utf-8")
    uce_ids = frame.iloc[:, 0].astype(str).tolist()
    formas = np.array(list(frame.columns[1:]))
    matriz = frame.iloc[:, 1:].to_numpy(dtype=np.int8)
    meta = pd.read_csv(prepared / "uce_meta.csv", sep=";", dtype=str, encoding="utf-8")
    gcol = "grupo" if "grupo" in meta.columns else "uci_id"
    gmap = dict(zip(meta["uce_id"].astype(str), meta[gcol].astype(str)))
    grupos = np.array([gmap.get(u, "?") for u in uce_ids])
    return matriz, formas, grupos


def _particao_subconjunto(
    matriz: np.ndarray,
    formas: np.ndarray,
    indices: np.ndarray,
    n_classes: int,
    min_freq: int,
) -> dict[int, int] | None:
    """Roda a CHD num subconjunto de UCEs com os filtros de run_saturation.

    Args:
        matriz: matriz documento-termo completa.
        formas: vetor de formas (colunas da matriz).
        indices: índices das UCEs incluídas no subconjunto.
        n_classes: número de classes da CHD.
        min_freq: frequência mínima da forma dentro do subconjunto.

    Returns:
        Dicionário índice de UCE para classe, ou None quando o subconjunto não
        comporta a classificação.
    """
    sub = matriz[indices]
    manter = sub.sum(axis=0) >= min_freq
    if manter.sum() < 2 or len(indices) < MIN_UCES_PARA_CHD:
        return None
    try:
        res = run_chd(sparse.csr_matrix(sub[:, manter]), list(formas[manter]),
                      [str(i) for i in indices], n_classes=n_classes)
    except Exception:
        return None
    return dict(zip((int(i) for i in res.uce_ids), (int(c) for c in res.assignments)))


def _ari_entre(pa: dict[int, int], pb: dict[int, int]) -> float | None:
    """Calcula o ARI entre duas partições nas UCEs comuns.

    Args:
        pa: primeira partição (índice de UCE para classe).
        pb: segunda partição.

    Returns:
        ARI nas UCEs comuns, ou None quando há menos que o mínimo de UCEs.
    """
    comuns = [i for i in pa if i in pb]
    if len(comuns) < MIN_UCES_COMUNS:
        return None
    return float(adjusted_rand_score([pa[i] for i in comuns],
                                     [pb[i] for i in comuns]))


def passos_cronologicos(
    matriz: np.ndarray,
    formas: np.ndarray,
    grupos: np.ndarray,
    n_classes: int,
    min_freq: int,
) -> list[dict]:
    """Reproduz os passos cronológicos de run_saturation.py.

    Args:
        matriz: matriz documento-termo completa.
        formas: vetor de formas.
        grupos: grupo de cada UCE.
        n_classes: número de classes da CHD.
        min_freq: frequência mínima por passo.

    Returns:
        Lista de dicionários por passo, com tamanho, vocabulário retido,
        ocorrências acumuladas, proporção de formas novas e ARI incremental.
    """
    invalidos = {"", "nan", "none", "?"}
    ordem = sorted(g for g in set(grupos.tolist()) if g and g.lower() not in invalidos)
    passos: list[dict] = []
    anterior_vocab: set[str] = set()
    anterior_part: dict[int, int] | None = None
    for k in range(1, len(ordem) + 1):
        mascara = np.isin(grupos, ordem[:k])
        indices = np.where(mascara)[0]
        sub = matriz[indices]
        manter = sub.sum(axis=0) >= min_freq
        vocab = set(formas[manter])
        particao = _particao_subconjunto(matriz, formas, indices, n_classes, min_freq)
        registro = {
            "k": k,
            "n_uce": int(mascara.sum()),
            "ocorrencias": int(sub.sum()),
            "vocab": len(vocab),
            "new_ratio": None,
            "chd_ari": None,
        }
        if anterior_vocab:
            novas = vocab - anterior_vocab
            registro["new_ratio"] = round(len(novas) / max(len(vocab), 1), 4)
        if anterior_part and particao:
            ari = _ari_entre(anterior_part, particao)
            registro["chd_ari"] = None if ari is None else round(ari, 4)
        passos.append(registro)
        anterior_vocab = vocab
        anterior_part = particao
    return passos


def estimar_estabilidade(
    matriz: np.ndarray,
    formas: np.ndarray,
    n_classes: int,
    min_freq: int,
    n_runs: int,
    seed: int,
    limiar_ari: float,
) -> dict:
    """Estima a distribuição do ARI de reamostras contra a partição completa.

    Args:
        matriz: matriz documento-termo completa.
        formas: vetor de formas.
        n_classes: número de classes da CHD.
        min_freq: frequência mínima por subconjunto.
        n_runs: número de reamostras.
        seed: semente do gerador.
        limiar_ari: limiar declarado do indicador (para a contagem de réplicas
            que o alcançam).

    Returns:
        Estatísticas da distribuição (média, desvio, percentis, extremos e a
        contagem de réplicas no limiar ou acima).
    """
    n = matriz.shape[0]
    referencia = _particao_subconjunto(matriz, formas, np.arange(n), n_classes,
                                       min_freq)
    rng = np.random.RandomState(seed)
    tamanho = int(SUBAMOSTRA_ESTABILIDADE * n)
    valores: list[float] = []
    for _ in range(n_runs):
        indices = np.sort(rng.choice(n, size=tamanho, replace=False))
        particao = _particao_subconjunto(matriz, formas, indices, n_classes,
                                         min_freq)
        if particao is None or referencia is None:
            continue
        ari = _ari_entre(referencia, particao)
        if ari is not None:
            valores.append(ari)
    arr = np.array(valores)
    return {
        "n_runs": int(len(arr)),
        "subamostra": SUBAMOSTRA_ESTABILIDADE,
        "ari_media": round(float(arr.mean()), 4),
        "ari_dp": round(float(arr.std(ddof=1)), 4),
        "ari_p2_5": round(float(np.percentile(arr, 2.5)), 4),
        "ari_p97_5": round(float(np.percentile(arr, 97.5)), 4),
        "ari_min": round(float(arr.min()), 4),
        "ari_max": round(float(arr.max()), 4),
        "replicas_no_limiar_ou_acima": int((arr >= limiar_ari).sum()),
        "limiar": limiar_ari,
    }


def linha_base_nula(
    matriz: np.ndarray,
    formas: np.ndarray,
    passos: list[dict],
    n_classes: int,
    min_freq: int,
    n_replicas: int,
    seed: int,
) -> list[dict]:
    """Computa a linha de base nula do ARI incremental por passo.

    Args:
        matriz: matriz documento-termo completa.
        formas: vetor de formas.
        passos: saída de passos_cronologicos.
        n_classes: número de classes da CHD.
        min_freq: frequência mínima por subconjunto.
        n_replicas: pares de subconjuntos aninhados por passo.
        seed: semente do gerador.

    Returns:
        Lista por passo com a distribuição nula e o percentil do observado.
    """
    n = matriz.shape[0]
    rng = np.random.RandomState(seed)
    resultado: list[dict] = []
    for k in range(1, len(passos)):
        n_prev = passos[k - 1]["n_uce"]
        n_atual = passos[k]["n_uce"]
        valores: list[float] = []
        for _ in range(n_replicas):
            idx_b = np.sort(rng.choice(n, size=n_atual, replace=False))
            idx_a = np.sort(rng.choice(idx_b, size=n_prev, replace=False))
            pa = _particao_subconjunto(matriz, formas, idx_a, n_classes, min_freq)
            pb = _particao_subconjunto(matriz, formas, idx_b, n_classes, min_freq)
            if pa is None or pb is None:
                continue
            ari = _ari_entre(pa, pb)
            if ari is not None:
                valores.append(ari)
        arr = np.array(valores)
        observado = passos[k]["chd_ari"]
        percentil = (None if observado is None
                     else round(float((arr < observado).mean() * 100), 1))
        resultado.append({
            "passo": k + 1,
            "n_prev": n_prev,
            "n_atual": n_atual,
            "ari_observado": observado,
            "ari_nulo_media": round(float(arr.mean()), 4),
            "ari_nulo_dp": round(float(arr.std(ddof=1)), 4),
            "ari_nulo_p2_5": round(float(np.percentile(arr, 2.5)), 4),
            "ari_nulo_p97_5": round(float(np.percentile(arr, 97.5)), 4),
            "percentil_observado_no_nulo": percentil,
            "n_replicas": int(len(arr)),
        })
    return resultado


def ajustar_heaps(passos: list[dict]) -> dict:
    """Ajusta V = K * N^beta em log-log e prediz a proporção de formas novas.

    Args:
        passos: saída de passos_cronologicos.

    Returns:
        Parâmetros do ajuste, R2 em log-log e as séries observada e predita.
    """
    ocorrencias = np.array([p["ocorrencias"] for p in passos], dtype=float)
    vocab = np.array([p["vocab"] for p in passos], dtype=float)
    log_n = np.log(ocorrencias)
    log_v = np.log(vocab)
    beta, log_k = np.polyfit(log_n, log_v, 1)
    predito = np.exp(log_k) * ocorrencias ** beta
    residuo = log_v - (log_k + beta * log_n)
    r2 = 1 - np.sum(residuo ** 2) / np.sum((log_v - log_v.mean()) ** 2)
    new_ratio_predito = [None] + [
        round(float((predito[i] - predito[i - 1]) / predito[i]), 4)
        for i in range(1, len(predito))
    ]
    return {
        "K": round(float(np.exp(log_k)), 3),
        "beta": round(float(beta), 4),
        "r2_loglog": round(float(r2), 4),
        "ocorrencias_acumuladas": [int(o) for o in ocorrencias],
        "vocab_observado": [int(v) for v in vocab],
        "vocab_predito": [round(float(v), 1) for v in predito],
        "new_ratio_observado": [p["new_ratio"] for p in passos],
        "new_ratio_predito_heaps": new_ratio_predito,
    }


def main() -> int:
    """Ponto de entrada da calibração."""
    parser = argparse.ArgumentParser(
        description="Calibração dos indicadores de saturação lexical")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--runs-estabilidade", type=int, default=100)
    parser.add_argument("--replicas-nula", type=int, default=50)
    parser.add_argument("--limiar-ari", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prepared = Path(args.prepared)
    saida = prepared / "saturation" / "calibracao"
    saida.mkdir(parents=True, exist_ok=True)

    matriz, formas, grupos = carregar_dados(prepared)
    print(f"{matriz.shape[0]} UCEs, {matriz.shape[1]} formas")

    passos = passos_cronologicos(matriz, formas, grupos, args.n_classes,
                                 args.min_freq)
    print("passos cronológicos reproduzidos:",
          [p["chd_ari"] for p in passos if p["chd_ari"] is not None])

    estabilidade = estimar_estabilidade(matriz, formas, args.n_classes,
                                        args.min_freq, args.runs_estabilidade,
                                        args.seed, args.limiar_ari)
    print(f"estabilidade: ARI médio {estabilidade['ari_media']} "
          f"[{estabilidade['ari_p2_5']}, {estabilidade['ari_p97_5']}], "
          f"{estabilidade['replicas_no_limiar_ou_acima']} réplica(s) no limiar")

    nula = linha_base_nula(matriz, formas, passos, args.n_classes,
                           args.min_freq, args.replicas_nula, args.seed + 1)
    for r in nula:
        print(f"passo {r['passo']}: obs {r['ari_observado']} | nulo "
              f"{r['ari_nulo_media']} [{r['ari_nulo_p2_5']}, "
              f"{r['ari_nulo_p97_5']}] | percentil {r['percentil_observado_no_nulo']}")

    heaps = ajustar_heaps(passos)
    print(f"Heaps: V = {heaps['K']} * N^{heaps['beta']} (R2 {heaps['r2_loglog']})")

    resultado = {
        "passos_cronologicos": passos,
        "estabilidade": estabilidade,
        "linha_base_nula": nula,
        "heaps": heaps,
        "parametros": {
            "n_classes": args.n_classes,
            "min_freq": args.min_freq,
            "runs_estabilidade": args.runs_estabilidade,
            "replicas_nula": args.replicas_nula,
            "limiar_ari": args.limiar_ari,
            "seed": args.seed,
        },
    }
    (saida / "calibracao_resultados.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(nula).to_csv(saida / "linha_base_nula.csv", sep=";", index=False,
                              encoding="utf-8")
    print(f"resultados em {saida}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
