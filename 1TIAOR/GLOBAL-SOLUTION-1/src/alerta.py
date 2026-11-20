"""
Geração do alerta para a vigilância em saúde e para a população.

O sistema APOIA a decisão; não substitui a equipe de vigilância epidemiológica.
"""
import numpy as np
import pandas as pd

import config
from semana_epi import somar_semanas
from tratamento import classificar_risco, colunas_features

RECOMENDACOES = {
    "baixo": [
        "Manter visitas de rotina dos agentes de endemias.",
        "Campanha educativa: eliminar água parada em vasos, pneus e calhas.",
    ],
    "medio": [
        "Intensificar visitas casa a casa nos bairros com mais notificações.",
        "Avisar UBS e UPAs para revisar estoque de soro e testes rápidos.",
        "Enviar alerta à população por SMS/WhatsApp/redes da prefeitura.",
    ],
    "alto": [
        "Acionar plano de contingência: mutirão de limpeza e bloqueio de transmissão.",
        "Ampliar horário de atendimento e montar polos de hidratação.",
        "Alerta máximo à população: procurar a UBS aos primeiros sintomas.",
    ],
}


def gerar_alerta(linha: pd.Series, classificador, regressor, geocode: int) -> dict:
    """Gera o alerta a partir da última semana disponível (linha de features)."""
    X = linha[colunas_features()].to_frame().T.astype(float)
    risco = str(classificador.predict(X)[0])
    probs = dict(zip(classificador.classes_, classificador.predict_proba(X)[0]))
    casos_prev = float(np.clip(np.expm1(regressor.predict(X)[0]), 0, None))

    return {
        "geocode": geocode,
        "municipio": config.MUNICIPIOS.get(geocode, str(geocode)),
        "se_referencia": int(linha["SE"]),
        "se_alvo": somar_semanas(int(linha["SE"]), config.HORIZONTE_SEMANAS),
        "risco": risco,
        "prob_alto": float(probs.get("alto", 0.0)),
        "probabilidades": {k: round(float(v), 3) for k, v in probs.items()},
        "casos_previstos": casos_prev,
        "risco_pela_regressao": classificar_risco(casos_prev / linha["pop"] * 1e5),
        "recomendacao": " | ".join(RECOMENDACOES[risco]),
    }


def montar_linha_cenario(casos_4_semanas: list[int], temps: list[float], umids: list[float],
                         semana: int, populacao: int) -> pd.Series:
    """
    Cria uma linha de features a partir de um cenário digitado pelo usuário
    (útil para demonstração e para testar "e se...?").
      casos_4_semanas: [hoje, 1 sem atrás, 2 sem atrás, 3 sem atrás]
      temps / umids  : [1, 2, 3, 4 semanas atrás]
    """
    if len(casos_4_semanas) != 4 or len(temps) != 4 or len(umids) != 4:
        raise ValueError("Informe 4 valores de casos, 4 de temperatura e 4 de umidade.")
    if any(c < 0 for c in casos_4_semanas):
        raise ValueError("Casos não podem ser negativos.")
    if not 1 <= semana <= 53:
        raise ValueError("Semana epidemiológica deve estar entre 1 e 53.")

    dados = {"SE": 202600 + semana, "pop": populacao}
    for lag, valor in zip(config.LAGS_CASOS, casos_4_semanas):
        dados[f"log_casos_lag{lag}"] = np.log1p(valor)
    dados["tendencia_2s"] = dados["log_casos_lag0"] - dados["log_casos_lag2"]
    for lag, t, u in zip(config.LAGS_CLIMA, temps, umids):
        dados[f"tempmed_lag{lag}"] = t
        dados[f"umidmed_lag{lag}"] = u
    dados["sen_semana"] = np.sin(2 * np.pi * semana / 52)
    dados["cos_semana"] = np.cos(2 * np.pi * semana / 52)
    return pd.Series(dados)


def formatar_alerta(alerta: dict) -> str:
    icone = {"baixo": "[OK]", "medio": "[ATENÇÃO]", "alto": "[ALERTA]"}[alerta["risco"]]
    linhas = [
        "=" * 64,
        f" {icone} SENTINELA DENGUE - {alerta['municipio']}",
        "=" * 64,
        f" Dados até a SE {alerta['se_referencia']}  ->  previsão para a SE {alerta['se_alvo']}",
        f" Risco previsto: {alerta['risco'].upper()}"
        f"  (prob. de risco alto: {alerta['prob_alto']:.0%})",
        f" Casos estimados na semana-alvo: {alerta['casos_previstos']:.0f}",
        " Ações recomendadas:",
    ]
    linhas += [f"   - {r}" for r in RECOMENDACOES[alerta["risco"]]]
    linhas.append(" * Apoio à decisão: não substitui a avaliação da vigilância em saúde.")
    linhas.append("=" * 64)
    return "\n".join(linhas)
