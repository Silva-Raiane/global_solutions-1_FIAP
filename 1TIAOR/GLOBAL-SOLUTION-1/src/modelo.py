"""
Modelos de Machine Learning (scikit-learn):

  * CLASSIFICAÇÃO -> risco (baixo / medio / alto) daqui a 2 semanas
  * REGRESSÃO     -> número de casos daqui a 2 semanas

Divisão TEMPORAL (não aleatória): treina com os anos mais antigos e testa com
os mais recentes, simulando o uso real (o modelo nunca "vê o futuro").
Todo modelo é comparado com uma linha de base ingênua (persistência:
"daqui a 2 semanas será igual a hoje"); só vale a pena usar ML se ganhar dela.
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, r2_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import analise
import config
from semana_epi import somar_semanas
from tratamento import colunas_features

ANO_CORTE_TESTE = 2023  # treino: até 2022 | teste: 2023 em diante


def dividir_temporal(tabela: pd.DataFrame, ano_corte: int = ANO_CORTE_TESTE):
    treino = tabela[tabela["ano"] < ano_corte]
    teste = tabela[tabela["ano"] >= ano_corte]
    if treino.empty or teste.empty:
        raise ValueError("Período insuficiente para a divisão treino/teste.")
    return treino, teste


def modelos_classificacao() -> dict:
    return {
        "Regressão Logística": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "Árvore de Decisão": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=config.SEMENTE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=config.SEMENTE
        ),
    }


def _metricas_class(y_real, y_prev) -> dict:
    return {
        "acuracia": accuracy_score(y_real, y_prev),
        "f1_macro": f1_score(y_real, y_prev, average="macro", labels=config.CLASSES_RISCO,
                             zero_division=0),
        "recall_alto": recall_score(y_real, y_prev, labels=["alto"], average="macro",
                                    zero_division=0),
    }


def treinar_classificacao(tabela: pd.DataFrame) -> dict:
    treino, teste = dividir_temporal(tabela)
    X_tr, y_tr = treino[colunas_features()], treino["alvo_risco"]
    X_te, y_te = teste[colunas_features()], teste["alvo_risco"]

    resultados = []
    # Linha de base: persistência (risco daqui a 2 semanas = risco de hoje)
    base_prev = teste["risco"]
    resultados.append({"modelo": "Linha de base (persistência)", **_metricas_class(y_te, base_prev)})

    treinados = {}
    for nome, modelo in modelos_classificacao().items():
        modelo.fit(X_tr, y_tr)
        prev = modelo.predict(X_te)
        treinados[nome] = modelo
        resultados.append({"modelo": nome, **_metricas_class(y_te, prev)})

    tabela_res = pd.DataFrame(resultados).round(3)
    candidatos = tabela_res[tabela_res["modelo"] != "Linha de base (persistência)"]
    # Critério: maior recall da classe "alto" (deixar passar uma epidemia custa
    # mais caro que um alarme falso); empate -> maior F1 macro
    melhor_nome = candidatos.sort_values(["recall_alto", "f1_macro"], ascending=False).iloc[0]["modelo"]
    melhor = treinados[melhor_nome]
    prev_melhor = melhor.predict(X_te)

    teste_prev = teste.assign(risco_previsto=prev_melhor)
    return {
        "tabela": tabela_res,
        "antecipacao": antecipacao_epidemias(teste_prev),
        "melhor_nome": melhor_nome,
        "melhor_modelo": melhor,
        "modelos_treinados": treinados,
        "matriz": confusion_matrix(y_te, prev_melhor, labels=config.CLASSES_RISCO),
        "teste": teste_prev,
        "n_treino": len(treino),
        "n_teste": len(teste),
    }


def antecipacao_epidemias(teste: pd.DataFrame) -> pd.DataFrame:
    """
    Pergunta prática da vigilância: "na semana em que o risco alto começou,
    o sistema já tinha avisado 2 semanas antes?"

    Para cada ano do teste com risco alto, pega a 1ª semana real de risco alto
    e verifica a previsão feita 2 semanas antes. A linha de base (persistência)
    nunca acerta esse início, porque só repete o risco que já está acontecendo.
    Também conta os alarmes falsos (previu alto e não foi alto) do ano.
    """
    linhas = []
    for ano, grupo in teste.groupby("ano"):
        inicio = grupo[(grupo["alvo_risco"] == "alto") & (grupo["risco"] != "alto")]
        if inicio.empty:
            continue
        primeira = inicio.iloc[0]
        se_inicio = somar_semanas(int(primeira["SE"]), config.HORIZONTE_SEMANAS)
        alarmes_falsos = int(((grupo["risco_previsto"] == "alto") & (grupo["alvo_risco"] != "alto")).sum())
        linhas.append({
            "ano": int(ano),
            "se_inicio_risco_alto": se_inicio,
            "alerta_emitido_na_SE": int(primeira["SE"]),
            "modelo_avisou_antes": "sim" if primeira["risco_previsto"] == "alto" else "não",
            "persistencia_avisou_antes": "não",
            "alarmes_falsos_no_ano": alarmes_falsos,
        })
    return pd.DataFrame(linhas)


def treinar_regressao(tabela: pd.DataFrame) -> dict:
    treino, teste = dividir_temporal(tabela)
    X_tr, X_te = treino[colunas_features()], teste[colunas_features()]

    modelo = make_pipeline(StandardScaler(), LinearRegression())
    modelo.fit(X_tr, treino["alvo_log_casos"])
    prev_casos = np.clip(np.expm1(modelo.predict(X_te)), 0, None)

    real = teste["alvo_casos"].to_numpy()
    base = teste["casos"].to_numpy()  # persistência
    tabela_res = pd.DataFrame([
        {"modelo": "Linha de base (persistência)", "MAE_casos": mean_absolute_error(real, base),
         "R2_log": r2_score(np.log1p(real), np.log1p(base))},
        {"modelo": "Regressão Linear (log)", "MAE_casos": mean_absolute_error(real, prev_casos),
         "R2_log": r2_score(np.log1p(real), np.log1p(prev_casos))},
    ]).round(3)
    return {"tabela": tabela_res, "modelo": modelo,
            "teste": teste.assign(casos_previstos=prev_casos)}


# ----------------------------------------------------------------------------
# Gráficos dos resultados
# ----------------------------------------------------------------------------
def grafico_previsao(teste: pd.DataFrame, sufixo: str = "") -> str:
    analise.estilo()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    datas = pd.to_datetime(teste["data_iniSE"]) + pd.Timedelta(weeks=config.HORIZONTE_SEMANAS)
    ax.plot(datas, teste["alvo_casos"], color=analise.AZUL, linewidth=1.8, label="Casos reais")
    ax.plot(datas, teste["casos_previstos"], color=analise.LARANJA, linewidth=1.8,
            linestyle="--", label="Previstos 2 semanas antes")
    ax.set_yscale("symlog", linthresh=50)
    ax.set_title("Período de teste: casos reais x previsão com 2 semanas de antecedência")
    ax.set_ylabel("casos por semana (escala log)")
    ax.legend(loc="upper right")
    return analise._salvar(fig, "05_previsao_teste", sufixo)


def grafico_matriz(matriz: np.ndarray, nome_modelo: str, sufixo: str = "") -> str:
    analise.estilo()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    rotulos = ["baixo", "médio", "alto"]
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=rotulos, yticklabels=rotulos, linewidths=2, linecolor="#fcfcfb")
    ax.set_xlabel("risco previsto")
    ax.set_ylabel("risco real")
    ax.set_title(f"Matriz de confusão: {nome_modelo}")
    ax.grid(False)
    return analise._salvar(fig, "06_matriz_confusao", sufixo)


def grafico_importancia(modelo, sufixo: str = "") -> str | None:
    if not hasattr(modelo, "feature_importances_"):
        return None
    analise.estilo()
    def legivel(nome: str) -> str:
        if nome.startswith("log_casos_lag"):
            n = nome[-1]
            return "casos desta semana" if n == "0" else f"casos há {n} sem."
        if nome.startswith("tempmed_lag"):
            return f"temperatura há {nome[-1]} sem."
        if nome.startswith("umidmed_lag"):
            return f"umidade há {nome[-1]} sem."
        return {"tendencia_2s": "tendência (2 sem.)", "sen_semana": "época do ano (seno)",
                "cos_semana": "época do ano (cosseno)"}.get(nome, nome)

    imp = pd.Series(modelo.feature_importances_, index=[legivel(c) for c in colunas_features()])
    imp = imp.sort_values().tail(10)
    config.PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    imp.sort_values(ascending=False).rename("importancia").to_csv(
        config.PASTA_RESULTADOS / f"importancia_variaveis{sufixo}.csv", index_label="variavel")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(imp.index, imp.values, color=analise.AZUL, height=0.6)
    ax.set_title("Variáveis mais importantes para prever o risco")
    ax.set_xlabel("importância relativa")
    ax.grid(axis="y", visible=False)
    return analise._salvar(fig, "07_importancia_variaveis", sufixo)


def salvar_modelos(class_res: dict, reg_res: dict) -> None:
    pasta = config.RAIZ / "src" / "modelos"
    pasta.mkdir(parents=True, exist_ok=True)
    joblib.dump(class_res["melhor_modelo"], pasta / "classificador_risco.joblib")
    joblib.dump(reg_res["modelo"], pasta / "regressor_casos.joblib")


def carregar_modelos():
    pasta = config.RAIZ / "src" / "modelos"
    return (joblib.load(pasta / "classificador_risco.joblib"),
            joblib.load(pasta / "regressor_casos.joblib"))


def rodar_modelos(tabela: pd.DataFrame, sufixo: str = "") -> dict:
    c = treinar_classificacao(tabela)
    r = treinar_regressao(tabela)
    figuras = [
        grafico_previsao(r["teste"], sufixo),
        grafico_matriz(c["matriz"], c["melhor_nome"], sufixo),
    ]
    # Importância das variáveis vem da Random Forest (modelo interpretável por árvore)
    imp = grafico_importancia(c["modelos_treinados"]["Random Forest"], sufixo)
    if imp:
        figuras.append(imp)
    salvar_modelos(c, r)
    config.PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    c["tabela"].to_csv(config.PASTA_RESULTADOS / f"metricas_classificacao{sufixo}.csv", index=False)
    r["tabela"].to_csv(config.PASTA_RESULTADOS / f"metricas_regressao{sufixo}.csv", index=False)
    c["antecipacao"].to_csv(config.PASTA_RESULTADOS / f"antecipacao_alertas{sufixo}.csv", index=False)
    return {"classificacao": c, "regressao": r, "figuras": figuras}
