"""
Banco de dados relacional (SQLite) do Sentinela Dengue.

Tabelas:
  municipio            -> municípios monitorados
  semana_epidemiologica-> série tratada (1 linha por município + semana)
  alerta               -> histórico de alertas emitidos pelo sistema

SQLite foi escolhido por não exigir servidor (um único arquivo .db) e por usar
SQL padrão, o que permite migrar para Oracle/PostgreSQL sem mudar a modelagem.
"""
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

import config

DDL = """
CREATE TABLE IF NOT EXISTS municipio (
    geocode     INTEGER PRIMARY KEY,
    nome        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS semana_epidemiologica (
    geocode     INTEGER NOT NULL REFERENCES municipio(geocode),
    se          INTEGER NOT NULL,          -- formato AAAASS
    data_inicio TEXT,
    casos       INTEGER NOT NULL CHECK (casos >= 0),
    incidencia  REAL    NOT NULL,          -- casos por 100 mil hab.
    tempmed     REAL,
    umidmed     REAL,
    risco       TEXT    CHECK (risco IN ('baixo','medio','alto')),
    base_simulada INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (geocode, se)
);

CREATE TABLE IF NOT EXISTS alerta (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    geocode          INTEGER NOT NULL REFERENCES municipio(geocode),
    se_referencia    INTEGER NOT NULL,     -- última semana com dados
    se_alvo          INTEGER NOT NULL,     -- semana prevista
    risco_previsto   TEXT    NOT NULL,
    prob_alto        REAL,
    casos_previstos  REAL,
    recomendacao     TEXT,
    gerado_em        TEXT    NOT NULL
);
"""


def conectar(caminho: Path | str = config.CAMINHO_BANCO) -> sqlite3.Connection:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(caminho)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(DDL)
    return con


def registrar_municipio(con, geocode: int, nome: str) -> None:
    con.execute("INSERT OR REPLACE INTO municipio (geocode, nome) VALUES (?, ?)", (geocode, nome))
    con.commit()


def gravar_semanas(con, df: pd.DataFrame, geocode: int, simulada: bool) -> int:
    """Insere ou atualiza (UPSERT) as semanas tratadas. Retorna quantas linhas."""
    linhas = [
        (geocode, int(r.SE), str(getattr(r, "data_iniSE", ""))[:10], int(r.casos),
         float(r.incidencia), float(r.tempmed), float(r.umidmed), r.risco, int(simulada))
        for r in df.itertuples()
    ]
    con.executemany(
        """INSERT INTO semana_epidemiologica
           (geocode, se, data_inicio, casos, incidencia, tempmed, umidmed, risco, base_simulada)
           VALUES (?,?,?,?,?,?,?,?,?)
           ON CONFLICT(geocode, se) DO UPDATE SET
             casos=excluded.casos, incidencia=excluded.incidencia,
             tempmed=excluded.tempmed, umidmed=excluded.umidmed,
             risco=excluded.risco, base_simulada=excluded.base_simulada""",
        linhas,
    )
    con.commit()
    return len(linhas)


def registrar_alerta(con, alerta: dict) -> int:
    cur = con.execute(
        """INSERT INTO alerta (geocode, se_referencia, se_alvo, risco_previsto,
                               prob_alto, casos_previstos, recomendacao, gerado_em)
           VALUES (?,?,?,?,?,?,?,?)""",
        (alerta["geocode"], alerta["se_referencia"], alerta["se_alvo"], alerta["risco"],
         alerta.get("prob_alto"), alerta.get("casos_previstos"), alerta.get("recomendacao"),
         datetime.now().isoformat(timespec="seconds")),
    )
    con.commit()
    return cur.lastrowid


# ----------------------------------------------------------------------------
# Consultas usadas no menu e no relatório
# ----------------------------------------------------------------------------
def consultar_alertas(con, limite: int = 10) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT a.id, m.nome AS municipio, a.se_referencia, a.se_alvo,
                  a.risco_previsto, ROUND(a.prob_alto, 3) AS prob_alto,
                  ROUND(a.casos_previstos) AS casos_previstos, a.gerado_em
           FROM alerta a JOIN municipio m ON m.geocode = a.geocode
           ORDER BY a.id DESC LIMIT ?""",
        con, params=(limite,),
    )


def semanas_por_risco(con, geocode: int) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT substr(se, 1, 4) AS ano,
                  SUM(risco = 'baixo') AS baixo,
                  SUM(risco = 'medio') AS medio,
                  SUM(risco = 'alto')  AS alto
           FROM semana_epidemiologica WHERE geocode = ?
           GROUP BY ano ORDER BY ano""",
        con, params=(geocode,),
    )


def piores_semanas(con, geocode: int, n: int = 5) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT se, data_inicio, casos, ROUND(incidencia, 1) AS incidencia,
                  ROUND(tempmed, 1) AS tempmed, risco
           FROM semana_epidemiologica WHERE geocode = ?
           ORDER BY casos DESC LIMIT ?""",
        con, params=(geocode, n),
    )
