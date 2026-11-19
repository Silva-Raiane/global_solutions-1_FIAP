"""
Leitura, limpeza e preparação dos dados para análise e Machine Learning.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config

COLUNAS_OBRIGATORIAS = ["SE", "casos", "tempmed", "umidmed"]
COLUNAS_CLIMA = ["tempmin", "tempmed", "tempmax", "umidmin", "umidmed", "umidmax"]


# ----------------------------------------------------------------------------
# Leitura
# ----------------------------------------------------------------------------
def localizar_base(geocode: int = config.GEOCODE_PADRAO) -> tuple[Path, bool]:
    """
    Procura primeiro a base REAL baixada do InfoDengue; se não existir,
    usa a base de demonstração SIMULADA.
    Retorna (caminho, eh_simulada).
    """
    real = config.PASTA_RAW / f"infodengue_{geocode}.csv"
    if real.exists():
        return real, False
    demo = config.PASTA_DEMO / "infodengue_demo_SIMULADO.csv"
    if demo.exists():
        return demo, True
    raise FileNotFoundError(
        "Nenhuma base encontrada. Rode 'python src/coleta_infodengue.py' (dados reais) "
        "ou 'python src/gerar_base_demo.py' (demonstração)."
    )


def carregar(caminho: Path | str) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {faltando}")
    return df


# ----------------------------------------------------------------------------
# Limpeza
# ----------------------------------------------------------------------------
def limpar(df: pd.DataFrame, populacao_padrao: int | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Aplica o tratamento e devolve (df_limpo, relatorio).
    O relatório registra o que foi feito, para entrar na documentação.
    """
    relatorio = {"linhas_entrada": len(df)}
    df = df.copy()

    # 1) Duplicatas de semana epidemiológica
    antes = len(df)
    df = df.drop_duplicates(subset="SE", keep="first")
    relatorio["duplicatas_removidas"] = antes - len(df)

    # 2) Ordem cronológica (a API entrega da mais recente para a mais antiga)
    df = df.sort_values("SE").reset_index(drop=True)
    if "data_iniSE" in df.columns:
        df["data_iniSE"] = pd.to_datetime(df["data_iniSE"])

    # 3) Casos negativos ou ausentes não fazem sentido -> descartados
    invalidos = df["casos"].isna() | (df["casos"] < 0)
    relatorio["casos_invalidos_removidos"] = int(invalidos.sum())
    df = df.loc[~invalidos].copy()
    df["casos"] = df["casos"].astype(int)

    # 4) Clima faltante: interpolação linear entre semanas vizinhas
    presentes = [c for c in COLUNAS_CLIMA if c in df.columns]
    relatorio["clima_faltante_interpolado"] = int(df[presentes].isna().sum().sum())
    df[presentes] = df[presentes].interpolate(limit_direction="both")

    # 5) População e incidência por 100 mil habitantes
    if "pop" not in df.columns or df["pop"].isna().all():
        if populacao_padrao is None:
            raise ValueError("Base sem coluna 'pop'; informe populacao_padrao.")
        df["pop"] = populacao_padrao
    df["pop"] = df["pop"].ffill().bfill()
    df["incidencia"] = df["casos"] / df["pop"] * 1e5

    df["ano"] = df["SE"] // 100
    df["semana"] = df["SE"] % 100
    df["risco"] = df["incidencia"].apply(classificar_risco)

    relatorio["linhas_saida"] = len(df)
    relatorio["periodo"] = f"{int(df['SE'].min())} a {int(df['SE'].max())}"
    return df.reset_index(drop=True), relatorio


def classificar_risco(incidencia: float) -> str:
    """Converte a incidência semanal (por 100 mil) em classe de risco."""
    if incidencia >= config.LIMIAR_ALTO:
        return "alto"
    elif incidencia >= config.LIMIAR_MEDIO:
        return "medio"
    return "baixo"


# ----------------------------------------------------------------------------
# Engenharia de atributos (features)
# ----------------------------------------------------------------------------
def criar_features(df: pd.DataFrame, horizonte: int = config.HORIZONTE_SEMANAS) -> pd.DataFrame:
    """
    Monta a tabela de treino: cada linha é uma semana 't' com informações
    disponíveis até 't' e o alvo da semana 't + horizonte'.
    """
    base = df.copy()

    # Casos recentes (escala log para reduzir o peso de semanas extremas)
    for lag in config.LAGS_CASOS:
        base[f"log_casos_lag{lag}"] = np.log1p(base["casos"].shift(lag))
    base["tendencia_2s"] = base["log_casos_lag0"] - base["log_casos_lag2"]

    # Clima defasado
    for lag in config.LAGS_CLIMA:
        base[f"tempmed_lag{lag}"] = base["tempmed"].shift(lag)
        base[f"umidmed_lag{lag}"] = base["umidmed"].shift(lag)

    # Sazonalidade (semana do ano como círculo, para SE 52 ficar perto da SE 1)
    base["sen_semana"] = np.sin(2 * np.pi * base["semana"] / 52)
    base["cos_semana"] = np.cos(2 * np.pi * base["semana"] / 52)

    # Alvos
    base["alvo_casos"] = base["casos"].shift(-horizonte)
    base["alvo_log_casos"] = np.log1p(base["alvo_casos"])
    base["alvo_risco"] = base["risco"].shift(-horizonte)

    return base


def colunas_features() -> list[str]:
    cols = [f"log_casos_lag{l}" for l in config.LAGS_CASOS] + ["tendencia_2s"]
    for l in config.LAGS_CLIMA:
        cols += [f"tempmed_lag{l}", f"umidmed_lag{l}"]
    cols += ["sen_semana", "cos_semana"]
    return cols


def tabela_treino(df_features: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas sem histórico suficiente ou sem alvo conhecido."""
    necessarias = colunas_features() + ["alvo_casos", "alvo_risco"]
    return df_features.dropna(subset=necessarias).reset_index(drop=True)


def salvar_processado(df: pd.DataFrame, nome: str = "dengue_tratado.csv") -> Path:
    config.PASTA_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    caminho = config.PASTA_PROCESSADOS / nome
    df.to_csv(caminho, index=False)
    return caminho
