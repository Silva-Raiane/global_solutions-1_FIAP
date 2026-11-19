"""
Gera uma BASE DE DEMONSTRAÇÃO SIMULADA com o mesmo formato da API do InfoDengue.

ATENÇÃO: os valores deste arquivo NÃO são dados reais de nenhum município.
Ela existe para:
  1) permitir rodar e testar o projeto sem internet;
  2) servir de base para os testes automatizados.

Como foi construída (para a banca entender as limitações):
  - semanas epidemiológicas de 2015 a 2025;
  - temperatura e umidade com sazonalidade típica do Sudeste (verão quente e úmido);
  - casos com pico anual no 1º semestre, intensidade diferente a cada ano
    (anos "epidêmicos" e anos "calmos") e efeito do clima de 2 a 3 semanas antes;
  - ruído aleatório, alguns valores faltantes e uma linha duplicada de propósito,
    para exercitar o tratamento de dados.

Uso:
    python src/gerar_base_demo.py
"""
import datetime as dt
import sys

import numpy as np
import pandas as pd

import config
from semana_epi import data_para_se as semana_epidemiologica
from semana_epi import inicio_semana_epi_1

POPULACAO_DEMO = 2_315_560  # ordem de grandeza de uma capital (valor de referência)

# Pico de incidência semanal (por 100 mil hab.) atribuído a cada ano SIMULADO
PICO_POR_ANO = {
    2015: 20, 2016: 180, 2017: 5, 2018: 4, 2019: 150, 2020: 15,
    2021: 4, 2022: 15, 2023: 60, 2024: 300, 2025: 40,
}


def gerar(semente: int = config.SEMENTE) -> pd.DataFrame:
    rng = np.random.default_rng(semente)

    inicio = inicio_semana_epi_1(config.ANO_INICIAL)
    fim = inicio_semana_epi_1(config.ANO_FINAL + 1)
    datas = []
    d = inicio
    while d < fim:
        datas.append(d)
        d += dt.timedelta(days=7)

    n = len(datas)
    se = np.array([semana_epidemiologica(x) for x in datas])
    semana = se % 100
    ano = se // 100
    angulo = 2 * np.pi * (semana - 6) / 52

    # Clima sazonal + ruído
    tempmed = 22.5 + 3.2 * np.cos(angulo) + rng.normal(0, 0.9, n)
    tempmin = tempmed - rng.uniform(4.5, 6.5, n)
    tempmax = tempmed + rng.uniform(5.0, 7.5, n)
    umidmed = 67 + 11 * np.cos(2 * np.pi * (semana - 3) / 52) + rng.normal(0, 4, n)
    umidmed = np.clip(umidmed, 35, 95)
    umidmin = np.clip(umidmed - rng.uniform(15, 25, n), 15, 90)
    umidmax = np.clip(umidmed + rng.uniform(10, 18, n), 40, 100)

    # Anomalias climáticas com defasagem (o mosquito responde semanas depois)
    anom_temp = pd.Series(tempmed - (22.5 + 3.2 * np.cos(angulo))).shift(3).fillna(0).to_numpy()
    anom_umid = pd.Series(umidmed - 67).shift(2).fillna(0).to_numpy()

    # Intensidade sazonal: curva em sino centrada perto da SE 13
    pico_ano = np.array([PICO_POR_ANO.get(a, 10) for a in ano], dtype=float)
    centro = 13 + rng.normal(0, 1.5, n)
    sino = np.exp(-((semana - centro) / 7.5) ** 2)
    incid_esperada = 0.6 + pico_ano * sino
    incid_esperada *= np.exp(0.22 * anom_temp + 0.012 * anom_umid)

    casos_esperados = incid_esperada * POPULACAO_DEMO / 1e5
    # Binomial negativa (super-dispersão típica de contagem de casos)
    k = 12
    p = k / (k + casos_esperados)
    casos = rng.negative_binomial(k, p)

    # Casos estimados: recentes recebem correção de atraso de notificação
    casos_est = casos * rng.uniform(1.0, 1.08, n)

    incid = casos / POPULACAO_DEMO * 1e5
    nivel = np.select([incid >= 50, incid >= 10, incid >= 3], [4, 3, 2], default=1)

    df = pd.DataFrame({
        "data_iniSE": [x.isoformat() for x in datas],
        "SE": se,
        "casos_est": casos_est.round(1),
        "casos": casos,
        "nivel": nivel,
        "pop": POPULACAO_DEMO,
        "tempmin": tempmin.round(2),
        "tempmed": tempmed.round(2),
        "tempmax": tempmax.round(2),
        "umidmin": umidmin.round(2),
        "umidmed": umidmed.round(2),
        "umidmax": umidmax.round(2),
    })

    # Imperfeições propositais para o tratamento de dados
    faltantes = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[faltantes, ["tempmed", "umidmed"]] = np.nan
    df = pd.concat([df, df.iloc[[100]]], ignore_index=True)  # linha duplicada

    # A API devolve da semana mais recente para a mais antiga
    return df.sort_values("SE", ascending=False).reset_index(drop=True)


def main() -> int:
    df = gerar()
    config.PASTA_DEMO.mkdir(parents=True, exist_ok=True)
    caminho = config.PASTA_DEMO / "infodengue_demo_SIMULADO.csv"
    df.to_csv(caminho, index=False)
    print(f"Base de demonstração SIMULADA: {len(df)} linhas em {caminho}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
