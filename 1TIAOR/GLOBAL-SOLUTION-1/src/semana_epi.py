"""
Funções de calendário epidemiológico (semanas epidemiológicas - SE).

A SE começa no domingo. A SE 01 de um ano é a semana que contém o dia 4 de
janeiro, por isso alguns anos têm 53 semanas (ex.: 2020 e 2025).
Código no formato AAAASS (ex.: 202414 = semana 14 de 2024).
"""
import datetime as dt


def inicio_semana_epi_1(ano: int) -> dt.date:
    """Domingo que inicia a SE 01 do ano."""
    quatro_jan = dt.date(ano, 1, 4)
    return quatro_jan - dt.timedelta(days=(quatro_jan.weekday() + 1) % 7)


def se_para_data(se: int) -> dt.date:
    """Converte AAAASS na data (domingo) de início da semana."""
    ano, semana = divmod(int(se), 100)
    return inicio_semana_epi_1(ano) + dt.timedelta(weeks=semana - 1)


def data_para_se(data: dt.date) -> int:
    """Converte uma data no código AAAASS da semana que a contém."""
    ano = data.year + 1
    while inicio_semana_epi_1(ano) > data:
        ano -= 1
    semana = (data - inicio_semana_epi_1(ano)).days // 7 + 1
    return ano * 100 + semana


def somar_semanas(se: int, semanas: int) -> int:
    """Avança 'semanas' a partir de uma SE, respeitando anos de 53 semanas."""
    return data_para_se(se_para_data(se) + dt.timedelta(weeks=semanas))
