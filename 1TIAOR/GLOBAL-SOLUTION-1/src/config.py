"""
Configurações centrais do projeto Sentinela Dengue.

Tudo que o grupo pode querer ajustar (município, período, limiares de risco,
caminhos de arquivos) fica aqui, para não espalhar "números mágicos" pelo código.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "data"
PASTA_RAW = PASTA_DADOS / "raw"          # dados reais baixados do InfoDengue
PASTA_DEMO = PASTA_DADOS / "demo"        # base de demonstração SIMULADA
PASTA_PROCESSADOS = PASTA_DADOS / "processed"
PASTA_FIGURAS = RAIZ / "docs" / "figuras"
PASTA_RESULTADOS = RAIZ / "docs" / "resultados"
CAMINHO_BANCO = PASTA_DADOS / "sentinela_dengue.db"

# ----------------------------------------------------------------------------
# Município monitorado (código IBGE de 7 dígitos)
# ----------------------------------------------------------------------------
MUNICIPIOS = {
    3106200: "Belo Horizonte - MG",
    3133808: "Itaúna - MG",
}
GEOCODE_PADRAO = 3106200

# Período padrão da coleta (anos epidemiológicos)
ANO_INICIAL = 2015
ANO_FINAL = 2025

# ----------------------------------------------------------------------------
# Previsão
# ----------------------------------------------------------------------------
# Quantas semanas à frente o sistema prevê o risco.
# 2 semanas = tempo mínimo para a vigilância organizar mutirão, visita
# domiciliar e reforço de atendimento.
HORIZONTE_SEMANAS = 2

# Defasagens (lags) usadas como variáveis explicativas
LAGS_CASOS = [0, 1, 2, 3]       # casos da semana atual e das 3 anteriores
LAGS_CLIMA = [1, 2, 3, 4]       # clima de 1 a 4 semanas atrás (ciclo do mosquito)

# ----------------------------------------------------------------------------
# Classes de risco (critério do projeto)
# ----------------------------------------------------------------------------
# Incidência semanal por 100 mil habitantes.
#   baixo : < 10
#   medio : 10 a < 50
#   alto  : >= 50
# Critério definido pelo grupo para a POC, inspirado na lógica de faixas de
# incidência usada na vigilância. Pode ser recalibrado com a equipe de saúde.
LIMIAR_MEDIO = 10.0
LIMIAR_ALTO = 50.0
CLASSES_RISCO = ["baixo", "medio", "alto"]

# Semente para reprodutibilidade
SEMENTE = 42
