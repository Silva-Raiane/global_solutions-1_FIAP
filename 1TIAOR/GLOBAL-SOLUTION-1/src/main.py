"""
SENTINELA DENGUE - programa principal.

Uso:
    python src/main.py              # menu interativo
    python src/main.py --pipeline   # roda tudo de uma vez (tratamento, análise, modelos, alerta)
"""
import argparse
import sys

import pandas as pd

import alerta as mod_alerta
import analise
import banco
import config
import modelo
import tratamento


def pipeline(geocode: int = config.GEOCODE_PADRAO, verbose: bool = True) -> dict:
    """Executa o fluxo completo e devolve os principais resultados."""
    caminho, simulada = tratamento.localizar_base(geocode)
    sufixo = "_SIMULADO" if simulada else ""
    origem = "BASE DE DEMONSTRAÇÃO SIMULADA" if simulada else "DADOS REAIS InfoDengue"
    log = print if verbose else (lambda *a, **k: None)

    log(f"\n[1/5] Lendo {caminho.name} ({origem})")
    bruto = tratamento.carregar(caminho)
    df, relatorio = tratamento.limpar(bruto)
    tratamento.salvar_processado(df, f"dengue_tratado{sufixo}.csv")
    log(f"      Tratamento: {relatorio}")

    log("[2/5] Gravando no banco SQLite")
    con = banco.conectar()
    nome = "Base demonstração (simulada)" if simulada else config.MUNICIPIOS.get(geocode, str(geocode))
    banco.registrar_municipio(con, geocode, nome)
    n = banco.gravar_semanas(con, df, geocode, simulada)
    log(f"      {n} semanas gravadas em {config.CAMINHO_BANCO.name}")

    log("[3/5] Análise exploratória e estatística")
    res_analise = analise.rodar_analise(df, sufixo)
    log(res_analise["resumo"].to_string())

    log("[4/5] Treinando modelos (treino até 2022, teste 2023 em diante)")
    tabela = tratamento.tabela_treino(tratamento.criar_features(df))
    res_modelos = modelo.rodar_modelos(tabela, sufixo)
    log("\nClassificação de risco (teste):")
    log(res_modelos["classificacao"]["tabela"].to_string(index=False))
    log(f"Modelo escolhido: {res_modelos['classificacao']['melhor_nome']}")
    log("\nO sistema avisou 2 semanas antes do início do risco alto?")
    log(res_modelos["classificacao"]["antecipacao"].to_string(index=False))
    log("\nRegressão de casos (teste):")
    log(res_modelos["regressao"]["tabela"].to_string(index=False))

    log("\n[5/5] Alerta para a semana mais recente")
    feats = tratamento.criar_features(df)
    ultima = feats.dropna(subset=tratamento.colunas_features()).iloc[-1]
    a = mod_alerta.gerar_alerta(ultima, res_modelos["classificacao"]["melhor_modelo"],
                                res_modelos["regressao"]["modelo"], geocode)
    if simulada:
        a["municipio"] = "BASE SIMULADA (demonstração)"
    banco.registrar_alerta(con, a)
    texto_alerta = mod_alerta.formatar_alerta(a)
    (config.PASTA_RESULTADOS / f"alerta_exemplo{sufixo}.txt").write_text(texto_alerta, encoding="utf-8")
    log(texto_alerta)
    con.close()

    return {"simulada": simulada, "relatorio": relatorio, "df": df, "analise": res_analise,
            "modelos": res_modelos, "alerta": a}


# ----------------------------------------------------------------------------
# Menu interativo
# ----------------------------------------------------------------------------
def ler_numero(msg: str, tipo=float, minimo=None, maximo=None):
    """Lê um número do teclado repetindo até ser válido."""
    while True:
        texto = input(msg).strip().replace(",", ".")
        try:
            valor = tipo(texto)
        except ValueError:
            print("  Valor inválido, tente de novo.")
            continue
        if (minimo is not None and valor < minimo) or (maximo is not None and valor > maximo):
            print(f"  Digite um valor entre {minimo} e {maximo}.")
            continue
        return valor


def simular_cenario() -> None:
    try:
        clf, reg = modelo.carregar_modelos()
    except FileNotFoundError:
        print("Treine os modelos primeiro (opção 1).")
        return
    print("\nCenário 'e se...?' - informe os dados do município:")
    semana = ler_numero("Semana epidemiológica atual (1-53): ", int, 1, 53)
    pop = ler_numero("População do município: ", int, 1000)
    casos = [ler_numero(f"Casos notificados {txt}: ", int, 0)
             for txt in ["nesta semana", "há 1 semana", "há 2 semanas", "há 3 semanas"]]
    temps, umids = [], []
    for s in range(1, 5):
        temps.append(ler_numero(f"Temperatura média há {s} semana(s) (°C): ", float, -5, 45))
        umids.append(ler_numero(f"Umidade média há {s} semana(s) (%): ", float, 0, 100))
    linha = mod_alerta.montar_linha_cenario(casos, temps, umids, semana, pop)
    a = mod_alerta.gerar_alerta(linha, clf, reg, geocode=0)
    a["municipio"] = "Cenário simulado pelo usuário"
    print(mod_alerta.formatar_alerta(a))


def menu() -> None:
    opcoes = {
        "1": "Rodar fluxo completo (tratar, analisar, treinar e emitir alerta)",
        "2": "Simular cenário 'e se...?'",
        "3": "Consultar últimos alertas (banco)",
        "4": "Consultar semanas por nível de risco (banco)",
        "5": "Consultar as 5 piores semanas (banco)",
        "0": "Sair",
    }
    while True:
        print("\n===== SENTINELA DENGUE =====")
        for k, v in opcoes.items():
            print(f" {k} - {v}")
        escolha = input("Opção: ").strip()

        if escolha == "1":
            pipeline()
        elif escolha == "2":
            simular_cenario()
        elif escolha in ("3", "4", "5"):
            con = banco.conectar()
            with pd.option_context("display.width", 140):
                if escolha == "3":
                    print(banco.consultar_alertas(con).to_string(index=False))
                elif escolha == "4":
                    print(banco.semanas_por_risco(con, config.GEOCODE_PADRAO).to_string(index=False))
                else:
                    print(banco.piores_semanas(con, config.GEOCODE_PADRAO).to_string(index=False))
            con.close()
        elif escolha == "0":
            print("Até logo!")
            break
        else:
            print("Opção inválida.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sentinela Dengue")
    parser.add_argument("--pipeline", action="store_true", help="roda o fluxo completo sem menu")
    parser.add_argument("--geocode", type=int, default=config.GEOCODE_PADRAO)
    args = parser.parse_args(argv)
    if args.pipeline:
        pipeline(args.geocode)
    else:
        menu()
    return 0


if __name__ == "__main__":
    sys.exit(main())
