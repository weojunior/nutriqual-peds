"""Sentimento pt-BR baseado em léxico, com tratamento de negação.

Aviso metodológico: este é um baseline transparente e auditável, NÃO uma
ferramenta clínica validada. Corrige dois problemas do código original:

* trata a NEGAÇÃO (não, nunca, sem, nem, jamais): inverte a polaridade dos
  próximos tokens dentro de uma janela ("não bom" deixa de contar como positivo);
* não baixa léxico remoto não verificado: o léxico é embutido e EDITÁVEL pelo
  usuário (parâmetro ``extra_lexicon``).

O léxico-semente é pequeno e cobre termos gerais; amplie-o para o seu domínio.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lexique import Lexique
from .preprocess import Preprocessor
from .tokenize import split_sentences, tokenize

#: Gatilhos de negação que invertem a polaridade dos tokens seguintes.
NEGATIONS: frozenset[str] = frozenset(
    {"não", "nao", "nunca", "jamais", "nem", "sem", "nenhum", "nenhuma", "tampouco"}
)

#: Alcance (em tokens) sobre o qual a negação inverte a polaridade.
NEGATION_SCOPE: int = 3

#: Léxico-semente pt-BR (lemas). Amplie conforme o seu corpus.
SEED_POSITIVE: frozenset[str] = frozenset({
    "bom", "ótimo", "otimo", "excelente", "feliz", "felicidade", "alegria", "amor",
    "esperança", "esperanca", "carinho", "apoio", "confiança", "confianca", "melhora",
    "melhorar", "saudável", "saudavel", "tranquilo", "calmo", "conforto", "confortável",
    "gostar", "adorar", "satisfeito", "satisfação", "satisfacao", "sucesso", "vitória",
    "vitoria", "cura", "curar", "forte", "força", "forca", "positivo", "agradável",
    "agradavel", "acolhimento", "gratidão", "gratidao", "paz", "união", "uniao", "fé", "fe",
})

#: Léxico-semente negativo (lemas).
SEED_NEGATIVE: frozenset[str] = frozenset({
    "ruim", "péssimo", "pessimo", "triste", "tristeza", "medo", "dor", "sofrer",
    "sofrimento", "angústia", "angustia", "ansiedade", "preocupação", "preocupacao",
    "preocupar", "cansaço", "cansaco", "cansado", "difícil", "dificil", "dificuldade",
    "raiva", "ódio", "odio", "desespero", "fraco", "fraqueza", "piorar", "pior",
    "doença", "doenca", "doente", "náusea", "nausea", "enjoo", "vômito", "vomito",
    "febre", "fadiga", "perda", "perder", "morte", "morrer", "chorar", "choro",
    "solidão", "solidao", "abandono", "negativo", "frustração", "frustracao", "culpa",
})


@dataclass
class SentimentResult:
    doc_id: str
    variables: dict[str, str]
    score: int          # soma de polaridade (+pos, -neg) com negação aplicada
    n_positive: int
    n_negative: int
    label: str          # "positivo" | "negativo" | "neutro"


def _polarity(lemma: str, positive: frozenset[str], negative: frozenset[str]) -> int:
    if lemma in positive:
        return 1
    if lemma in negative:
        return -1
    return 0


def score_text(
    text: str,
    processor: "Preprocessor | Lexique",
    positive: frozenset[str] = SEED_POSITIVE,
    negative: frozenset[str] = SEED_NEGATIVE,
) -> tuple[int, int, int]:
    """Pontua um texto; retorna (score, n_positivos, n_negativos) com negação."""
    pre = processor if isinstance(processor, Preprocessor) else Preprocessor(processor)
    score = n_pos = n_neg = 0
    for sentence in split_sentences(text):
        tokens = tokenize(sentence)
        negate_until = -1
        for pos, token in enumerate(tokens):
            if token in NEGATIONS:
                negate_until = pos + NEGATION_SCOPE
                continue
            lemma = pre.lemma(token)
            pol = _polarity(lemma, positive, negative)
            if pol == 0:
                continue
            if pos <= negate_until:
                pol = -pol            # inverte sob escopo de negação
            if pol > 0:
                n_pos += 1
            else:
                n_neg += 1
            score += pol
    return score, n_pos, n_neg


def label_for(score: int) -> str:
    if score > 0:
        return "positivo"
    if score < 0:
        return "negativo"
    return "neutro"
