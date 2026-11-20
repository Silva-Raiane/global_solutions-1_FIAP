"""
Coleta dos dados REAIS de dengue no InfoDengue (FIOCRUZ / FGV).

API pública, sem cadastro:
    https://info.dengue.mat.br/api/alertcity

Cada linha da resposta é uma semana epidemiológica (SE) do município, com
casos notificados, casos estimados (nowcasting), temperatura, umidade,
população e o nível de alerta do próprio InfoDengue.

Uso:
    python src/coleta_infodengue.py                    # Belo Horizonte, 2015-2025
    python src/coleta_infodengue.py --geocode 3133808  # Itaúna
"""
import argparse
import io
import sys
import urllib.parse
import urllib.request

import pandas as pd

import config

URL_API = "https://info.dengue.mat.br/api/alertcity"


def montar_url(geocode: int, ano_ini: int, ano_fim: int, doenca: str = "dengue") -> str:
    """Monta a URL da API com os parâmetros de consulta."""
    parametros = {
        "geocode": geocode,
        "disease": doenca,
        "format": "csv",
        "ew_start": 1,
        "ew_end": 53,
        "ey_start": ano_ini,
        "ey_end": ano_fim,
    }
    return f"{URL_API}?{urllib.parse.urlencode(parametros)}"


def baixar_dados(geocode: int, ano_ini: int, ano_fim: int) -> pd.DataFrame:
    """Baixa a série semanal do município e devolve um DataFrame."""
    url = montar_url(geocode, ano_ini, ano_fim)
    print(f"Baixando: {url}")
    with urllib.request.urlopen(url, timeout=60) as resposta:
        conteudo = resposta.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(conteudo))
    if df.empty:
        raise ValueError("A API respondeu sem linhas. Confira o geocode e o período.")
    return df


def salvar(df: pd.DataFrame, geocode: int) -> str:
    config.PASTA_RAW.mkdir(parents=True, exist_ok=True)
    caminho = config.PASTA_RAW / f"infodengue_{geocode}.csv"
    df.to_csv(caminho, index=False)
    return str(caminho)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Coleta dados reais do InfoDengue")
    parser.add_argument("--geocode", type=int, default=config.GEOCODE_PADRAO)
    parser.add_argument("--inicio", type=int, default=config.ANO_INICIAL)
    parser.add_argument("--fim", type=int, default=config.ANO_FINAL)
    args = parser.parse_args(argv)

    try:
        df = baixar_dados(args.geocode, args.inicio, args.fim)
    except Exception as erro:  # rede, API fora do ar, etc.
        print(f"Falha na coleta: {erro}")
        print("Sem internet? Use a base de demonstração: python src/gerar_base_demo.py")
        return 1

    caminho = salvar(df, args.geocode)
    print(f"{len(df)} semanas salvas em {caminho}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
