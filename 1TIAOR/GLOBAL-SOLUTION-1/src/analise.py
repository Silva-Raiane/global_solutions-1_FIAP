"""
Análise exploratória e estatística descritiva (Pandas, NumPy, Matplotlib, Seaborn).
"""
import matplotlib

matplotlib.use("Agg")  # gera imagens sem precisar de janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

# Paleta (validada para daltonismo) e tinta de texto
AZUL = "#2a78d6"
LARANJA = "#eb6834"
CINZA_GRADE = "#e1e0d9"
CINZA_EIXO = "#c3c2b7"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
CORES_RISCO = {"baixo": "#0ca30c", "medio": "#fab219", "alto": "#d03b3b"}


def estilo():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb", "axes.edgecolor": CINZA_EIXO,
        "axes.grid": True, "grid.color": CINZA_GRADE, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
        "text.color": TEXTO, "font.size": 10, "legend.frameon": False,
    })


def _salvar(fig, nome: str, sufixo: str = "") -> str:
    config.PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    caminho = config.PASTA_FIGURAS / f"{nome}{sufixo}.png"
    fig.tight_layout()
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)
    return str(caminho)


# ----------------------------------------------------------------------------
# Estatística descritiva
# ----------------------------------------------------------------------------
def resumo_estatistico(df: pd.DataFrame) -> pd.DataFrame:
    """Média, mediana, desvio-padrão, mínimo, máximo e quartis das variáveis-chave."""
    cols = ["casos", "incidencia", "tempmed", "umidmed"]
    tabela = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    tabela = tabela.rename(columns={"mean": "media", "std": "desvio", "50%": "mediana"})
    return tabela[["count", "media", "mediana", "desvio", "min", "25%", "75%", "max"]].round(2)


def casos_por_ano(df: pd.DataFrame) -> pd.DataFrame:
    tabela = df.groupby("ano").agg(
        casos_total=("casos", "sum"),
        pico_semanal=("casos", "max"),
        semanas_risco_alto=("risco", lambda s: int((s == "alto").sum())),
        temp_media=("tempmed", "mean"),
    )
    tabela["incidencia_anual_100k"] = (
        tabela["casos_total"] / df.groupby("ano")["pop"].mean() * 1e5
    ).round(1)
    tabela["temp_media"] = tabela["temp_media"].round(2)
    return tabela


def correlacao_clima(df: pd.DataFrame, max_lag: int = 8) -> pd.DataFrame:
    """
    Correlação de Spearman entre o clima de 'lag' semanas atrás e os casos de hoje.
    Spearman é usado porque a relação não é linear e os casos têm cauda longa.
    """
    linhas = []
    for lag in range(0, max_lag + 1):
        linhas.append({
            "lag_semanas": lag,
            "temperatura": df["casos"].corr(df["tempmed"].shift(lag), method="spearman"),
            "umidade": df["casos"].corr(df["umidmed"].shift(lag), method="spearman"),
        })
    return pd.DataFrame(linhas).round(3)


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
def grafico_serie(df: pd.DataFrame, sufixo: str = "") -> str:
    estilo()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = df["data_iniSE"] if "data_iniSE" in df.columns else df.index
    ax.plot(x, df["incidencia"], color=AZUL, linewidth=1.6)
    for limiar, nome in [(config.LIMIAR_MEDIO, "médio"), (config.LIMIAR_ALTO, "alto")]:
        ax.axhline(limiar, color=TEXTO_2, linewidth=0.9, linestyle="--")
        ax.annotate(f"limiar {nome} ({limiar:g})", xy=(1.0, limiar), xycoords=("axes fraction", "data"),
                    xytext=(4, -3), textcoords="offset points", fontsize=8, color=TEXTO_2)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(bottom=0)
    ax.set_title("Incidência semanal de dengue (casos por 100 mil hab.)")
    ax.set_ylabel("incidência (escala log)")
    return _salvar(fig, "01_serie_incidencia", sufixo)


def grafico_sazonalidade(df: pd.DataFrame, sufixo: str = "") -> str:
    estilo()
    perfil = df.groupby("semana")["incidencia"].agg(["median", lambda s: s.quantile(0.9)])
    perfil.columns = ["mediana", "p90"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.fill_between(perfil.index, 0, perfil["p90"], color=AZUL, alpha=0.15, linewidth=0)
    ax.plot(perfil.index, perfil["mediana"], color=AZUL, linewidth=2)
    ax.annotate("mediana", xy=(perfil["mediana"].idxmax(), perfil["mediana"].max()),
                xytext=(8, 0), textcoords="offset points", color=TEXTO_2, fontsize=9)
    ax.annotate("90% das semanas abaixo", xy=(perfil["p90"].idxmax(), perfil["p90"].max()),
                xytext=(8, -4), textcoords="offset points", color=TEXTO_2, fontsize=9)
    ax.set_title("Sazonalidade: incidência por semana epidemiológica (todos os anos)")
    ax.set_xlabel("semana epidemiológica")
    ax.set_ylabel("incidência por 100 mil")
    ax.set_xlim(1, 53)
    ax.set_ylim(bottom=0)
    return _salvar(fig, "02_sazonalidade", sufixo)


def grafico_anos(tabela_anos: pd.DataFrame, sufixo: str = "") -> str:
    estilo()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    barras = ax.bar(tabela_anos.index.astype(str), tabela_anos["incidencia_anual_100k"],
                    color=AZUL, width=0.6)
    ax.bar_label(barras, fmt="%.0f", fontsize=8, color=TEXTO_2, padding=2)
    ax.set_title("Incidência anual (casos por 100 mil hab.)")
    ax.grid(axis="x", visible=False)
    return _salvar(fig, "03_incidencia_anual", sufixo)


def grafico_correlacao(corr: pd.DataFrame, sufixo: str = "") -> str:
    estilo()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    largura = 0.38
    x = np.arange(len(corr))
    ax.bar(x - largura / 2 - 0.01, corr["temperatura"], largura, color=AZUL, label="Temperatura média")
    ax.bar(x + largura / 2 + 0.01, corr["umidade"], largura, color=LARANJA, label="Umidade média")
    ax.axhline(0, color=CINZA_EIXO, linewidth=1)
    ax.set_xticks(x, [f"{l}" for l in corr["lag_semanas"]])
    ax.set_xlabel("defasagem (semanas entre o clima e os casos)")
    ax.set_ylabel("correlação de Spearman")
    ax.set_title("Clima de semanas anteriores x casos de dengue")
    ax.legend(loc="upper left", ncols=2)
    ax.set_ylim(top=max(1.0, float(corr[["temperatura", "umidade"]].max().max()) + 0.15))
    ax.grid(axis="x", visible=False)
    return _salvar(fig, "04_correlacao_clima", sufixo)


def rodar_analise(df: pd.DataFrame, sufixo: str = "") -> dict:
    resumo = resumo_estatistico(df)
    anos = casos_por_ano(df)
    corr = correlacao_clima(df)
    figuras = [
        grafico_serie(df, sufixo),
        grafico_sazonalidade(df, sufixo),
        grafico_anos(anos, sufixo),
        grafico_correlacao(corr, sufixo),
    ]
    config.PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(config.PASTA_RESULTADOS / f"resumo_estatistico{sufixo}.csv")
    anos.to_csv(config.PASTA_RESULTADOS / f"casos_por_ano{sufixo}.csv")
    corr.to_csv(config.PASTA_RESULTADOS / f"correlacao_clima{sufixo}.csv", index=False)
    return {"resumo": resumo, "anos": anos, "correlacao": corr, "figuras": figuras}
