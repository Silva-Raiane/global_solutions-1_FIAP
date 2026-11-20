"""
Testes automatizados do Sentinela Dengue (pytest).

Rodar a partir da raiz do projeto:
    python -m pytest -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import alerta  # noqa: E402
import banco  # noqa: E402
import config  # noqa: E402
import gerar_base_demo  # noqa: E402
import modelo  # noqa: E402
import semana_epi  # noqa: E402
import tratamento  # noqa: E402


@pytest.fixture(scope="module")
def base_demo():
    return gerar_base_demo.gerar()


@pytest.fixture(scope="module")
def tratada(base_demo):
    df, relatorio = tratamento.limpar(base_demo)
    return df, relatorio


# --------------------------------------------------------------------------
# Calendário epidemiológico
# --------------------------------------------------------------------------
def test_semana_epi_conhecidas():
    import datetime as dt
    # SE 01/2024 começou em 31/12/2023 (domingo)
    assert semana_epi.data_para_se(dt.date(2023, 12, 31)) == 202401
    assert semana_epi.se_para_data(202401) == dt.date(2023, 12, 31)


def test_soma_semanas_vira_o_ano_com_53_semanas():
    assert semana_epi.somar_semanas(202553, 2) == 202602   # 2025 tem 53 SE
    assert semana_epi.somar_semanas(202452, 2) == 202502   # 2024 tem 52 SE


# --------------------------------------------------------------------------
# Tratamento
# --------------------------------------------------------------------------
@pytest.mark.parametrize("incidencia,esperado", [
    (0, "baixo"), (9.99, "baixo"), (10, "medio"), (49.9, "medio"), (50, "alto"), (300, "alto"),
])
def test_classificar_risco_limiares(incidencia, esperado):
    assert tratamento.classificar_risco(incidencia) == esperado


def test_limpeza_remove_duplicata_e_ordena(tratada):
    df, rel = tratada
    assert rel["duplicatas_removidas"] == 1
    assert df["SE"].is_unique
    assert df["SE"].is_monotonic_increasing


def test_limpeza_interpola_clima(tratada):
    df, rel = tratada
    assert rel["clima_faltante_interpolado"] > 0
    assert df[["tempmed", "umidmed"]].isna().sum().sum() == 0


def test_limpeza_descarta_casos_negativos():
    bruto = pd.DataFrame({"SE": [202401, 202402, 202403], "casos": [5, -1, 7],
                          "tempmed": [25, 26, 27], "umidmed": [70, 71, 72], "pop": [1000] * 3})
    df, rel = tratamento.limpar(bruto)
    assert rel["casos_invalidos_removidos"] == 1
    assert list(df["casos"]) == [5, 7]


def test_colunas_obrigatorias(tmp_path):
    caminho = tmp_path / "ruim.csv"
    pd.DataFrame({"SE": [1], "casos": [1]}).to_csv(caminho, index=False)
    with pytest.raises(ValueError):
        tratamento.carregar(caminho)


def test_features_nao_usam_o_futuro(tratada):
    """O alvo é a semana t+2; nenhuma feature pode conter informação de t+1 ou t+2."""
    df, _ = tratada
    feats = tratamento.criar_features(df)
    i = 200
    assert feats.loc[i, "alvo_casos"] == df.loc[i + config.HORIZONTE_SEMANAS, "casos"]
    assert feats.loc[i, "log_casos_lag0"] == pytest.approx(np.log1p(df.loc[i, "casos"]))
    assert feats.loc[i, "tempmed_lag1"] == df.loc[i - 1, "tempmed"]


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tabela(tratada):
    return tratamento.tabela_treino(tratamento.criar_features(tratada[0]))


def test_divisao_temporal_sem_sobreposicao(tabela):
    treino, teste = modelo.dividir_temporal(tabela)
    assert treino["SE"].max() < teste["SE"].min()


def test_classificador_detecta_risco_alto(tabela):
    res = modelo.treinar_classificacao(tabela)
    escolhido = res["tabela"].set_index("modelo").loc[res["melhor_nome"]]
    base = res["tabela"].set_index("modelo").loc["Linha de base (persistência)"]
    # Requisito do projeto: detectar risco alto melhor que "repetir a semana atual"
    assert escolhido["recall_alto"] >= base["recall_alto"]
    assert set(res["teste"]["risco_previsto"]) <= set(config.CLASSES_RISCO)


def test_regressao_supera_linha_de_base(tabela):
    res = modelo.treinar_regressao(tabela).get("tabela").set_index("modelo")
    assert res.loc["Regressão Linear (log)", "MAE_casos"] < res.loc["Linha de base (persistência)", "MAE_casos"]


# --------------------------------------------------------------------------
# Alerta
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def modelos_treinados(tabela):
    c = modelo.treinar_classificacao(tabela)
    r = modelo.treinar_regressao(tabela)
    return c["melhor_modelo"], r["modelo"]


def test_cenario_epidemico_gera_alerta_alto(modelos_treinados):
    clf, reg = modelos_treinados
    # casos subindo rápido no verão quente e úmido (incidência ~ 70/100 mil)
    linha = alerta.montar_linha_cenario([1600, 1100, 700, 450], [26, 26, 25.5, 25],
                                        [80, 79, 78, 78], semana=10, populacao=2_315_560)
    a = alerta.gerar_alerta(linha, clf, reg, geocode=0)
    assert a["risco"] == "alto"


def test_cenario_calmo_gera_alerta_baixo(modelos_treinados):
    clf, reg = modelos_treinados
    linha = alerta.montar_linha_cenario([15, 14, 16, 15], [19, 19, 18.5, 19],
                                        [55, 56, 55, 57], semana=32, populacao=2_315_560)
    a = alerta.gerar_alerta(linha, clf, reg, geocode=0)
    assert a["risco"] == "baixo"


def test_cenario_invalido_e_recusado():
    with pytest.raises(ValueError):
        alerta.montar_linha_cenario([-1, 0, 0, 0], [20] * 4, [60] * 4, 10, 1000)
    with pytest.raises(ValueError):
        alerta.montar_linha_cenario([1, 1, 1], [20] * 4, [60] * 4, 10, 1000)


# --------------------------------------------------------------------------
# Banco de dados
# --------------------------------------------------------------------------
def test_banco_upsert_e_consultas(tmp_path, tratada):
    df, _ = tratada
    con = banco.conectar(tmp_path / "teste.db")
    banco.registrar_municipio(con, 1, "Teste")
    banco.gravar_semanas(con, df, 1, simulada=True)
    banco.gravar_semanas(con, df, 1, simulada=True)  # 2ª vez não duplica
    total = con.execute("SELECT COUNT(*) FROM semana_epidemiologica").fetchone()[0]
    assert total == len(df)

    por_risco = banco.semanas_por_risco(con, 1)
    assert por_risco[["baixo", "medio", "alto"]].to_numpy().sum() == len(df)

    id_alerta = banco.registrar_alerta(con, {"geocode": 1, "se_referencia": 202401,
                                             "se_alvo": 202403, "risco": "alto"})
    assert id_alerta == 1
    assert banco.consultar_alertas(con).iloc[0]["risco_previsto"] == "alto"
    con.close()


def test_banco_recusa_risco_invalido(tmp_path):
    import sqlite3
    con = banco.conectar(tmp_path / "t.db")
    banco.registrar_municipio(con, 1, "Teste")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO semana_epidemiologica (geocode, se, casos, incidencia, risco) "
                    "VALUES (1, 202401, 5, 1.0, 'gravissimo')")
    con.close()
