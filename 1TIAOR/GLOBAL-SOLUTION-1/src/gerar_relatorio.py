"""
Gera o documento de entrega da Global Solution (HTML pronto para virar PDF).

Os números, tabelas, gráficos, trechos de código e a saída dos testes são lidos
dos arquivos gerados pelo próprio projeto. Assim, depois de rodar com os dados
REAIS do InfoDengue, basta executar de novo este script para atualizar tudo.

Uso:
    python src/main.py --pipeline        # gera figuras e métricas
    python src/gerar_relatorio.py        # gera docs/relatorio_entrega.html
Depois: abra o HTML no navegador -> Imprimir (Ctrl+P) -> Salvar como PDF.
"""
import ast
import base64
import html
import subprocess
import sys
from datetime import date

import pandas as pd

import config

# ----------------------------------------------------------------------------
# Dados da capa - EDITAR AQUI
# ----------------------------------------------------------------------------
GRUPO = "Grupo Sentinela"
INTEGRANTES = [
    ("Raiane Sirlene da Silva", "RM575984"),
    ("Rodrigo Textor de Textor", "RM574243"),
    ("Pedro Victor Pires", "RM575181"),
]
TURMA = "1TIAOR 2026/2"
LINK_VIDEO = "COLOCAR AQUI O LINK DO VÍDEO NO YOUTUBE (NÃO LISTADO)"
LINK_REPO = "COLOCAR AQUI O LINK DO REPOSITÓRIO GITHUB PRIVADO"

SRC = config.RAIZ / "src"
MESES = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto",
         "Setembro", "Outubro", "Novembro", "Dezembro"]


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def esc(t) -> str:
    return html.escape(str(t))


def detectar_sufixo() -> str:
    """Usa resultados reais se existirem; senão, os da base simulada."""
    if (config.PASTA_RESULTADOS / "metricas_classificacao.csv").exists():
        return ""
    return "_SIMULADO"


def ler(nome: str, sufixo: str, **kw) -> pd.DataFrame:
    return pd.read_csv(config.PASTA_RESULTADOS / f"{nome}{sufixo}.csv", **kw)


def figura(nome: str, sufixo: str, legenda: str, n: int, largura: str = "100%") -> str:
    caminho = config.PASTA_FIGURAS / f"{nome}{sufixo}.png"
    if not caminho.exists():
        return f"<p class='aviso'>Figura {esc(nome)} não encontrada.</p>"
    b64 = base64.b64encode(caminho.read_bytes()).decode()
    return (f"<figure><img src='data:image/png;base64,{b64}' style='width:{largura}'>"
            f"<figcaption>Figura {n}. {legenda}</figcaption></figure>")


def num_br(x: float) -> str:
    """Formata número no padrão brasileiro (1.234,5)."""
    texto = f"{x:,.3f}".rstrip("0").rstrip(".")
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def tabela_html(df: pd.DataFrame, legenda: str, n: int, index: bool = False) -> str:
    t = df.to_html(index=index, border=0, classes="tab", float_format=num_br)
    return f"<p class='leg-tab'>Tabela {n}. {legenda}</p>{t}"


def trecho(arquivo: str, nome: str) -> str:
    """Extrai o código-fonte de uma função (ou constante) de um arquivo."""
    fonte = (SRC / arquivo).read_text(encoding="utf-8")
    arvore = ast.parse(fonte)
    for no in arvore.body:
        if isinstance(no, (ast.FunctionDef, ast.ClassDef)) and no.name == nome:
            return ast.get_source_segment(fonte, no)
        if isinstance(no, ast.Assign) and any(getattr(t, "id", None) == nome for t in no.targets):
            return ast.get_source_segment(fonte, no)
    raise KeyError(f"{nome} não encontrado em {arquivo}")


def bloco_codigo(arquivo: str, nome: str, titulo: str) -> str:
    return (f"<p class='cod-tit'><b>{esc(titulo)}</b> <span>src/{esc(arquivo)} · {esc(nome)}</span></p>"
            f"<pre class='code'>{esc(trecho(arquivo, nome))}</pre>")


def rodar_testes() -> str:
    r = subprocess.run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider"],
                       cwd=config.RAIZ, capture_output=True, text=True)
    linhas = [l for l in r.stdout.splitlines() if "::" in l or "passed" in l or "failed" in l]
    return "\n".join(l.replace("tests/test_sentinela.py::", "") for l in linhas)


# ----------------------------------------------------------------------------
# Montagem
# ----------------------------------------------------------------------------
def montar() -> str:
    suf = detectar_sufixo()
    simulado = suf == "_SIMULADO"

    resumo = ler("resumo_estatistico", suf, index_col=0)
    anos = ler("casos_por_ano", suf)
    corr = ler("correlacao_clima", suf)
    mclass = ler("metricas_classificacao", suf)
    mreg = ler("metricas_regressao", suf)
    antec = ler("antecipacao_alertas", suf)
    alerta_txt = (config.PASTA_RESULTADOS / f"alerta_exemplo{suf}.txt").read_text(encoding="utf-8")
    testes_txt = rodar_testes()
    n_testes = sum(1 for l in testes_txt.splitlines() if "PASSED" in l)
    n_falhas = sum(1 for l in testes_txt.splitlines() if "FAILED" in l)

    import tratamento
    import modelo as mod
    caminho, _ = tratamento.localizar_base()
    bruto = tratamento.carregar(caminho)
    df, rel = tratamento.limpar(bruto)
    tab = tratamento.tabela_treino(tratamento.criar_features(df))
    treino, teste = mod.dividir_temporal(tab)

    base = mclass[mclass["modelo"].str.startswith("Linha")].iloc[0]
    cand = mclass[~mclass["modelo"].str.startswith("Linha")]
    melhor = cand.sort_values(["recall_alto", "f1_macro"], ascending=False).iloc[0]
    rb = mreg[mreg["modelo"].str.startswith("Linha")].iloc[0]
    rl = mreg[~mreg["modelo"].str.startswith("Linha")].iloc[0]
    ganho_mae = (1 - rl["MAE_casos"] / rb["MAE_casos"]) * 100
    avisos = int((antec["modelo_avisou_antes"] == "sim").sum())
    temporadas = len(antec)
    alarmes = int(antec["alarmes_falsos_no_ano"].sum())
    lag_temp = corr.loc[corr["temperatura"].idxmax()]
    if melhor["f1_macro"] < base["f1_macro"]:
        texto_f1 = ("Isso é esperado: na maior parte das semanas o risco de daqui a 2 semanas é igual ao de hoje, "
                    "e a persistência acerta essas semanas \"fáceis\". O valor do modelo está nas <b>mudanças de "
                    "nível</b>, que é justamente o momento em que a prefeitura precisa agir.")
    else:
        texto_f1 = ("Ou seja, o modelo supera a regra ingênua também no desempenho geral, além de antecipar as "
                    "<b>mudanças de nível</b>, que é o momento em que a prefeitura precisa agir.")
    imp = ler("importancia_variaveis", suf)
    top_imp = ", ".join(f"<i>{esc(v)}</i>" for v in imp["variavel"].head(4))
    municipio = "a base de demonstração simulada" if simulado else config.MUNICIPIOS.get(config.GEOCODE_PADRAO)

    def cmp(a, b, maior_melhor=True):
        if abs(a - b) < 1e-9:
            return "igual ao"
        return "acima do" if (a > b) == maior_melhor else "abaixo do"

    aviso_sim = (
        "<div class='aviso'><b>Atenção:</b> os resultados desta seção foram obtidos com a "
        "<b>BASE DE DEMONSTRAÇÃO SIMULADA</b> (arquivo <code>infodengue_demo_SIMULADO.csv</code>), "
        "que reproduz o formato da API do InfoDengue mas não representa nenhum município real. "
        "Eles demonstram que o pipeline funciona de ponta a ponta. Os valores com dados reais "
        "são obtidos rodando <code>python src/coleta_infodengue.py</code> e o pipeline novamente.</div>"
        if simulado else ""
    )

    nf = iter(range(1, 50))
    nt = iter(range(1, 50))

    integrantes_html = "".join(f"<tr><td>{esc(n)}</td><td>{esc(rm)}</td></tr>" for n, rm in INTEGRANTES)

    anos_fmt = anos.rename(columns={
        "ano": "Ano", "casos_total": "Casos", "pico_semanal": "Pico semanal",
        "semanas_risco_alto": "Semanas risco alto", "temp_media": "Temp. média (°C)",
        "incidencia_anual_100k": "Incidência anual /100 mil"})
    resumo_fmt = resumo.rename(index={"casos": "Casos/semana", "incidencia": "Incidência/100 mil",
                                      "tempmed": "Temperatura média (°C)", "umidmed": "Umidade média (%)"},
                               columns={"count": "n", "media": "Média", "mediana": "Mediana",
                                        "desvio": "Desvio-padrão", "min": "Mín.", "max": "Máx."})
    mclass_fmt = mclass.rename(columns={"modelo": "Modelo", "acuracia": "Acurácia",
                                        "f1_macro": "F1 macro", "recall_alto": "Recall (risco alto)"})
    mreg_fmt = mreg.rename(columns={"modelo": "Modelo", "MAE_casos": "Erro médio absoluto (casos)",
                                    "R2_log": "R² (escala log)"})
    antec_fmt = antec.rename(columns={
        "ano": "Ano", "se_inicio_risco_alto": "SE início risco alto",
        "alerta_emitido_na_SE": "Previsão feita na SE", "modelo_avisou_antes": "Sentinela avisou?",
        "persistencia_avisou_antes": "Linha de base avisou?", "alarmes_falsos_no_ano": "Alarmes falsos"})
    corr_fmt = corr.rename(columns={"lag_semanas": "Defasagem (semanas)", "temperatura": "Temperatura",
                                    "umidade": "Umidade"})

    tecnologias = pd.DataFrame([
        ["Python 3", "Linguagem do curso; funções, decisões, laços, leitura de arquivos", "Todo o projeto"],
        ["Pandas / NumPy", "Leitura do CSV, limpeza, agregações, defasagens e cálculos", "tratamento.py, analise.py"],
        ["Matplotlib / Seaborn", "Gráficos de série, sazonalidade, correlação e resultados", "analise.py, modelo.py"],
        ["scikit-learn", "Classificação e regressão introdutórias, métricas", "modelo.py"],
        ["SQLite + SQL", "Banco relacional sem servidor, com chaves e restrições", "banco.py"],
        ["pytest", "Testes automatizados que comprovam o funcionamento", "tests/"],
        ["API InfoDengue", "Dados oficiais, abertos e semanais, já com clima", "coleta_infodengue.py"],
    ], columns=["Tecnologia", "Por que foi usada", "Onde"])

    disciplinas = pd.DataFrame([
        ["Lógica e programação em Python", "Funções, if/elif/else, laços while/for, validação de entrada no menu, leitura de CSV, módulos"],
        ["Estatística", "Média, mediana, desvio-padrão, quartis, correlação de Spearman, sazonalidade"],
        ["Análise e tratamento de dados", "Duplicatas, ordenação, valores inválidos, interpolação, engenharia de atributos"],
        ["Machine Learning", "Classificação (Regressão Logística, Árvore, Random Forest), regressão linear, divisão temporal, métricas e linha de base"],
        ["Banco de dados", "Modelagem relacional, DDL, chaves primária e estrangeira, CHECK, UPSERT, JOIN e GROUP BY"],
        ["Visualização e comunicação", "Gráficos acessíveis, tabelas de resultado, alerta em linguagem simples"],
    ], columns=["Disciplina / conteúdo", "Como aparece no projeto"])

    variaveis = pd.DataFrame([
        ["SE", "Semana epidemiológica (AAAASS)"], ["data_iniSE", "Data de início da semana"],
        ["casos", "Casos notificados na semana"], ["casos_est", "Casos estimados (correção de atraso)"],
        ["tempmin/tempmed/tempmax", "Temperatura mínima, média e máxima (°C)"],
        ["umidmin/umidmed/umidmax", "Umidade relativa mínima, média e máxima (%)"],
        ["pop", "População do município"], ["nivel", "Nível de alerta do próprio InfoDengue (1 a 4)"],
    ], columns=["Coluna", "Significado"])

    css = """
    @page { size: A4; margin: 2cm 2cm 2cm 2cm; }
    body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 11pt; color: #1b1b1b; line-height: 1.45; }
    h1 { font-size: 17pt; color: #0d366b; border-bottom: 2px solid #2a78d6; padding-bottom: 4px;
         page-break-before: always; margin-top: 0; }
    h2 { font-size: 13pt; color: #184f95; margin-top: 18px; page-break-after: avoid; }
    p { text-align: justify; margin: 6px 0; }
    .capa { text-align: center; height: 24cm; display: flex; flex-direction: column; justify-content: space-between; }
    .capa h1 { page-break-before: avoid; border: none; font-size: 24pt; color: #0d366b; }
    .capa p { text-align: center; }
    .capa .sub { font-size: 13pt; color: #52514e; }
    .capa table { margin: 0 auto; border-collapse: collapse; font-size: 12pt; }
    .capa td { padding: 4px 16px; border-bottom: 1px solid #e1e0d9; text-align: left; }
    .tab { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 4px 0 12px; page-break-inside: avoid; }
    .tab th { background: #e8f0fb; color: #0d366b; text-align: left; padding: 5px 6px; border-bottom: 1px solid #9ec5f4; }
    .tab td { padding: 4px 6px; border-bottom: 1px solid #e1e0d9; }
    .leg-tab { font-size: 9.5pt; color: #52514e; margin: 10px 0 2px; font-weight: bold; page-break-after: avoid; }
    figure { margin: 10px 0 14px; page-break-inside: avoid; text-align: center; }
    figcaption { font-size: 9.5pt; color: #52514e; margin-top: 4px; }
    pre.code, pre.saida { font-family: Consolas, 'Courier New', monospace; font-size: 8.2pt; line-height: 1.3;
         background: #f6f7f9; border: 1px solid #e1e0d9; border-left: 3px solid #2a78d6; padding: 8px 10px;
         white-space: pre-wrap; word-break: break-word; }
    pre.saida { border-left-color: #0ca30c; }
    .cod-tit { margin: 12px 0 2px; font-size: 10pt; page-break-after: avoid; }
    .cod-tit span { color: #52514e; font-family: Consolas, monospace; font-size: 8.5pt; }
    .aviso { background: #fff7e0; border: 1px solid #fab219; padding: 8px 10px; font-size: 10pt; margin: 8px 0; }
    .destaque { background: #eef5ff; border-left: 4px solid #2a78d6; padding: 8px 12px; margin: 10px 0; }
    .fluxo { display: flex; gap: 6px; align-items: stretch; margin: 10px 0; page-break-inside: avoid; }
    .fluxo div.passo { flex: 1; border: 1px solid #9ec5f4; background: #f3f8fe; padding: 6px; font-size: 8.8pt; text-align: center; border-radius: 4px; }
    .fluxo div.passo b { display: block; color: #0d366b; font-size: 9.5pt; margin-bottom: 2px; }
    .fluxo div.seta { align-self: center; color: #2a78d6; font-weight: bold; }
    ul { margin: 4px 0 8px 18px; padding: 0; } li { margin: 2px 0; }
    .sumario li { list-style: none; }
    code { font-family: Consolas, monospace; font-size: 9.5pt; background: #f1f1ef; padding: 0 3px; }
    .links td { padding: 6px 8px; }
    """

    passos = [
        ("1. Coleta", "API InfoDengue → CSV semanal"),
        ("2. Tratamento", "Pandas: limpeza, incidência, defasagens"),
        ("3. Banco", "SQLite: semanas e alertas"),
        ("4. Análise", "Estatística e gráficos"),
        ("5. Modelos", "Classificação de risco e regressão de casos"),
        ("6. Alerta", "Risco em 2 semanas + ações recomendadas"),
    ]
    fluxo = "<div class='fluxo'>" + "<div class='seta'>→</div>".join(
        f"<div class='passo'><b>{esc(a)}</b>{esc(b)}</div>" for a, b in passos) + "</div>"

    H = []
    # ------------------------------------------------------------------ CAPA
    H.append(f"""
    <div class='capa'>
      <div>
        <p class='sub'>FIAP – Faculdade de Informática e Administração Paulista<br>
        Graduação em Inteligência Artificial</p>
        <p class='sub'><b>Global Solution 2026.2 · CitizenTech – Tecnologia centrada no cidadão</b></p>
      </div>
      <div>
        <h1>Sentinela Dengue</h1>
        <p class='sub'>Alerta antecipado de risco de dengue com Inteligência Artificial e dados abertos</p>
      </div>
      <div>
        <p><b>{esc(GRUPO)}</b> · Turma <b>{esc(TURMA)}</b></p>
        <table><tr><td><b>Integrante</b></td><td><b>RM</b></td></tr>{integrantes_html}</table>
      </div>
      <div class='sub'>{MESES[date.today().month - 1]} de {date.today().year}</div>
    </div>""")

    # ------------------------------------------------------------- SUMÁRIO
    H.append("""<h1>Sumário</h1><ul class='sumario'>
      <li>1. Introdução</li><li>2. Desenvolvimento</li>
      <li style='margin-left:16px'>2.1 Visão geral da solução · 2.2 Tecnologias · 2.3 Dados · 2.4 Tratamento ·
      2.5 Análise estatística · 2.6 Modelagem · 2.7 Banco de dados · 2.8 Alerta e menu ·
      2.9 Integração entre disciplinas · 2.10 Códigos principais</li>
      <li>3. Resultados obtidos</li><li>4. Impacto e Objetivos de Desenvolvimento Sustentável</li>
      <li>5. Limitações e próximos passos</li><li>6. Conclusões</li><li>7. Links e referências</li></ul>""")

    # ---------------------------------------------------------- INTRODUÇÃO
    H.append(f"""
    <h1>1. Introdução</h1>
    <h2>1.1 Problema e contexto</h2>
    <p>A dengue é a arbovirose de maior impacto no Brasil e se repete todo ano com picos no verão e no outono,
    quando calor e chuva favorecem a reprodução do mosquito <i>Aedes aegypti</i>. O problema que o grupo escolheu
    não é a falta de dados, e sim o <b>tempo de reação</b>: em geral a alta de casos só fica evidente quando as
    unidades de saúde já estão lotadas. Mutirões de limpeza, visitas casa a casa, reforço de equipes, compra de
    insumos e campanhas de aviso levam dias para serem organizados.</p>
    <p>Esses dados já existem e são públicos. O sistema InfoDengue (FIOCRUZ e FGV) publica, toda semana e para
    cada município, os casos notificados e o clima. Falta uma ferramenta simples que transforme essa série em um
    aviso antecipado e em ações concretas.</p>
    <h2>1.2 Público beneficiado</h2>
    <ul>
      <li><b>Equipes de vigilância epidemiológica e gestores municipais de saúde</b>, que recebem o alerta e a
      lista de ações sugeridas;</li>
      <li><b>Moradores</b>, em especial idosos, crianças e pessoas que dependem do SUS, que podem ser avisados
      antes do pico e procurar atendimento logo nos primeiros sintomas;</li>
      <li><b>Unidades de saúde</b> (UBS e UPAs), que podem se preparar com insumos e escalas.</li>
    </ul>
    <h2>1.3 Objetivo</h2>
    <div class='destaque'>Prever, com <b>2 semanas de antecedência</b>, o nível de risco de dengue (baixo, médio ou
    alto) e o número esperado de casos em um município, usando dados abertos e técnicas introdutórias de
    Machine Learning, e transformar essa previsão em um alerta com ações recomendadas.</div>
    <p>O sistema é uma ferramenta de <b>apoio à decisão</b> e não substitui a avaliação dos profissionais de
    vigilância em saúde.</p>
    <h2>1.4 Objetivos de Desenvolvimento Sustentável relacionados</h2>
    <ul>
      <li><b>ODS 3 – Saúde e bem-estar:</b> meta 3.3 (combater doenças transmissíveis) e meta 3.d (reforçar a
      capacidade de alerta precoce e de gestão de riscos à saúde);</li>
      <li><b>ODS 11 – Cidades e comunidades sustentáveis:</b> cidades mais seguras e resilientes, com dados
      abertos orientando a ação do poder público nos bairros.</li>
    </ul>""")

    # ------------------------------------------------------- DESENVOLVIMENTO
    H.append(f"""
    <h1>2. Desenvolvimento</h1>
    <h2>2.1 Visão geral da solução</h2>
    <p>O Sentinela Dengue é um programa em Python organizado em módulos. O fluxo completo roda com um único
    comando (<code>python src/main.py --pipeline</code>) e também pode ser usado por um menu interativo.</p>
    {fluxo}
    <h2>2.2 Tecnologias e justificativa</h2>
    {tabela_html(tecnologias, "Tecnologias utilizadas e justificativa.", next(nt))}
    <h2>2.3 Origem dos dados</h2>
    <p><b>Fonte real:</b> API pública do InfoDengue (<code>https://info.dengue.mat.br/api/alertcity</code>), sem
    necessidade de cadastro. A consulta é feita pelo código IBGE do município (padrão: Belo Horizonte, 3106200;
    Itaúna, 3133808, também está configurada) para as semanas epidemiológicas de {config.ANO_INICIAL} a
    {config.ANO_FINAL}. Cada linha é uma semana.</p>
    {tabela_html(variaveis, "Principais colunas utilizadas.", next(nt))}
    <p><b>Base de demonstração:</b> como a coleta depende de internet e da disponibilidade da API, o projeto inclui
    uma base de demonstração <b>simulada</b>, gerada por <code>src/gerar_base_demo.py</code> com o mesmo formato
    da API: sazonalidade de clima típica do Sudeste, anos epidêmicos e anos calmos, efeito do clima com defasagem
    de 2 a 3 semanas e ruído aleatório. Ela também contém, de propósito, valores faltantes e uma linha duplicada,
    para exercitar o tratamento. Ela <b>não representa nenhum município real</b>, e todo arquivo gerado a partir
    dela recebe o sufixo <code>_SIMULADO</code>.</p>
    <p><b>Limitações dos dados reais:</b> casos notificados dependem da procura por atendimento e da notificação
    (há subnotificação); as semanas mais recentes chegam incompletas por atraso de notificação; o clima é um valor
    único para todo o município; não há informação por bairro na API utilizada.</p>

    <h2>2.4 Tratamento dos dados</h2>
    <p>Fonte usada nesta execução: <b>{esc(municipio)}</b>, período {esc(rel['periodo'])}.</p>
    <ul>
      <li>Linhas recebidas: <b>{rel['linhas_entrada']}</b>; duplicatas de semana removidas: <b>{rel['duplicatas_removidas']}</b>;</li>
      <li>registros com casos ausentes ou negativos descartados: <b>{rel['casos_invalidos_removidos']}</b>;</li>
      <li>valores de clima faltantes preenchidos por interpolação linear entre semanas vizinhas: <b>{rel['clima_faltante_interpolado']}</b>;</li>
      <li>semanas ordenadas cronologicamente (a API devolve da mais recente para a mais antiga);</li>
      <li>cálculo da <b>incidência</b> (casos por 100 mil habitantes), que permite comparar municípios de tamanhos diferentes;</li>
      <li>classificação de cada semana em risco baixo (&lt; {config.LIMIAR_MEDIO:g}), médio ({config.LIMIAR_MEDIO:g} a
      &lt; {config.LIMIAR_ALTO:g}) ou alto (≥ {config.LIMIAR_ALTO:g}), critério definido pelo grupo para a POC e
      ajustável em <code>config.py</code>.</li>
    </ul>
    <p>Linhas finais: <b>{rel['linhas_saida']}</b> semanas.</p>

    <h2>2.5 Análise estatística</h2>
    {aviso_sim}
    {tabela_html(resumo_fmt.drop(columns=['n']), "Estatística descritiva das variáveis principais (por semana).", next(nt), index=True)}
    <p>A média semanal de casos ({num_br(round(resumo.loc['casos','media']))}) é muito maior que a mediana
    ({num_br(round(resumo.loc['casos','mediana']))}). Isso indica uma distribuição assimétrica: na maior parte do ano há poucos
    casos e, em poucas semanas, há picos muito altos. Por isso o projeto trabalha com escala logarítmica e usa a
    correlação de Spearman, que não exige relação linear.</p>
    {figura("01_serie_incidencia", suf, "Incidência semanal com os limiares de risco médio e alto (escala logarítmica).", next(nf))}
    {figura("02_sazonalidade", suf, "Perfil sazonal: mediana da incidência por semana epidemiológica e faixa até o percentil 90.", next(nf))}
    {figura("03_incidencia_anual", suf, "Incidência anual por 100 mil habitantes.", next(nf))}
    {tabela_html(anos_fmt, "Resumo por ano.", next(nt))}
    {figura("04_correlacao_clima", suf, "Correlação de Spearman entre o clima de semanas anteriores e os casos da semana.", next(nf))}
    <p>A correlação entre temperatura e casos cresce quando se compara o clima de semanas anteriores
    (máxima de {num_br(lag_temp['temperatura'])} com {int(lag_temp['lag_semanas'])} semanas de defasagem). Isso é
    coerente com o ciclo do mosquito e com o período de incubação. Parte dessa correlação vem da própria
    sazonalidade (verão quente e com mais casos), por isso os modelos usam o clima defasado <i>junto</i> com a
    época do ano e com os casos recentes, e não isoladamente.</p>
    {tabela_html(corr_fmt, "Correlação de Spearman por defasagem.", next(nt))}

    <h2>2.6 Modelagem com Machine Learning</h2>
    <p><b>Variáveis de entrada</b> (tudo que é conhecido na semana <i>t</i>): casos da semana e das 3 anteriores
    (em log), tendência das últimas 2 semanas, temperatura e umidade médias de 1 a 4 semanas antes e a época do ano
    (seno e cosseno da semana, para que a SE 52 fique "perto" da SE 1).</p>
    <p><b>Alvos</b> (semana <i>t + 2</i>): o nível de risco (<b>classificação</b>) e o número de casos (<b>regressão</b>).</p>
    <p><b>Divisão temporal:</b> treino com {len(treino)} semanas (SE {int(treino['SE'].min())} a {int(treino['SE'].max())})
    e teste com {len(teste)} semanas (SE {int(teste['SE'].min())} a {int(teste['SE'].max())}). A divisão não é aleatória
    para simular o uso real: o modelo aprende com o passado e é avaliado em anos que nunca viu.</p>
    <p><b>Modelos comparados:</b> Regressão Logística (com padronização), Árvore de Decisão (profundidade 4) e
    Random Forest para a classificação; Regressão Linear sobre o log dos casos para a regressão. Os modelos de
    classificação usam <code>class_weight="balanced"</code>, porque semanas de risco alto são minoria.</p>
    <p><b>Linha de base:</b> todo modelo é comparado com a regra ingênua "daqui a 2 semanas será igual a hoje"
    (persistência). Usar ML só se justifica se ele trouxer algo que essa regra não traz.</p>
    <p><b>Critério de escolha:</b> maior <i>recall</i> da classe "alto", ou seja, a proporção de semanas de risco
    alto que o modelo conseguiu prever. Deixar passar uma epidemia custa mais caro que um alarme falso. Em caso de
    empate, vence o maior F1 macro.</p>

    <h2>2.7 Banco de dados</h2>
    <p>A série tratada e o histórico de alertas ficam em um banco relacional SQLite
    (<code>data/sentinela_dengue.db</code>) com três tabelas: <b>municipio</b>, <b>semana_epidemiologica</b>
    (chave primária composta município + semana, restrições CHECK para casos não negativos e risco válido) e
    <b>alerta</b> (cada alerta emitido, com data e hora). A gravação usa UPSERT, então rodar o pipeline de novo
    atualiza as semanas sem duplicá-las. O menu oferece consultas SQL com JOIN e GROUP BY.</p>
    {bloco_codigo("banco.py", "DDL", "Criação das tabelas (DDL)")}

    <h2>2.8 Alerta e menu interativo</h2>
    <p>Ao final do pipeline, o sistema usa a semana mais recente para prever a semana <i>t + 2</i> e emite um alerta
    com o risco previsto, a probabilidade de risco alto, os casos estimados e as ações recomendadas para cada nível.
    O alerta é gravado no banco. O menu permite rodar o fluxo, simular um cenário "e se...?" digitando casos e clima
    (com validação das entradas em laço) e consultar o banco.</p>

    <h2>2.9 Integração entre disciplinas</h2>
    {tabela_html(disciplinas, "Conteúdos da Fase 1 à Fase 3 aplicados no projeto.", next(nt))}

    <h2>2.10 Códigos principais</h2>
    <p>Os trechos abaixo foram extraídos automaticamente do código-fonte. O código completo está no repositório.</p>
    {bloco_codigo("tratamento.py", "limpar", "Limpeza e tratamento")}
    {bloco_codigo("tratamento.py", "classificar_risco", "Classificação da semana em nível de risco")}
    {bloco_codigo("tratamento.py", "criar_features", "Engenharia de atributos (defasagens e alvos)")}
    {bloco_codigo("modelo.py", "dividir_temporal", "Divisão temporal treino/teste")}
    {bloco_codigo("modelo.py", "treinar_classificacao", "Treino e comparação dos classificadores")}
    {bloco_codigo("alerta.py", "gerar_alerta", "Geração do alerta")}
    {bloco_codigo("main.py", "ler_numero", "Leitura validada do teclado (menu)")}
    """)

    # ------------------------------------------------------------ RESULTADOS
    H.append(f"""
    <h1>3. Resultados obtidos</h1>
    {aviso_sim}
    <h2>3.1 Classificação do risco em 2 semanas</h2>
    {tabela_html(mclass_fmt, "Desempenho no período de teste.", next(nt))}
    <p>O modelo escolhido foi a <b>{esc(melhor['modelo'])}</b>, com recall de risco alto de
    <b>{melhor['recall_alto']:.0%}</b>, {cmp(melhor['recall_alto'], base['recall_alto'])} valor da linha de base
    ({base['recall_alto']:.0%}). Em acurácia ({num_br(melhor['acuracia'] * 100)}% contra {num_br(base['acuracia'] * 100)}%)
    e F1 macro ({num_br(melhor['f1_macro'])} contra {num_br(base['f1_macro'])}), o modelo ficou
    {cmp(melhor['f1_macro'], base['f1_macro'])} valor da linha de base. {texto_f1}</p>
    {figura("06_matriz_confusao", suf, "Matriz de confusão do modelo escolhido no período de teste.", next(nf), "62%")}

    <h2>3.2 O sistema avisou antes do risco alto começar?</h2>
    {tabela_html(antec_fmt, "Início do período de risco alto em cada ano de teste.", next(nt))}
    <p>Em <b>{avisos} de {temporadas}</b> temporadas com risco alto no período de teste, o Sentinela previu o início do
    risco alto <b>2 semanas antes</b>. A linha de base não consegue fazer isso, porque só repete o risco atual.
    Em troca, o modelo emitiu <b>{alarmes}</b> alarmes falsos de risco alto no período (semanas previstas como alto
    que ficaram em médio ou baixo). Esse é o equilíbrio escolhido: preferir avisar a mais do que deixar passar.</p>

    <h2>3.3 Previsão do número de casos</h2>
    {tabela_html(mreg_fmt, "Regressão dos casos em 2 semanas, período de teste.", next(nt))}
    <p>A regressão linear teve erro médio absoluto de <b>{num_br(round(rl['MAE_casos']))}</b> casos por semana, contra
    {num_br(round(rb['MAE_casos']))} da linha de base ({'redução' if ganho_mae >= 0 else 'aumento'} de {num_br(round(abs(ganho_mae), 1))}%),
    e R² de {num_br(rl['R2_log'])} na escala logarítmica.</p>
    {figura("05_previsao_teste", suf, "Casos reais e casos previstos 2 semanas antes, no período de teste.", next(nf))}
    {figura("07_importancia_variaveis", suf, "Variáveis mais importantes segundo a Random Forest.", next(nf), "80%")}
    <p>As quatro variáveis que mais pesam na Random Forest são, em ordem: {top_imp} (Figura 7). O histórico recente de casos indica a
    tendência da transmissão, e o clima defasado e a época do ano ajudam a antecipar as viradas de nível.</p>

    <h2>3.4 Evidências de funcionamento</h2>
    <p><b>Alerta emitido pelo sistema</b> (saída real do programa, gravada também no banco):</p>
    <pre class='saida'>{esc(alerta_txt)}</pre>
    <p><b>Testes automatizados</b> ({n_testes} aprovados, {n_falhas} com falha). Eles verificam o calendário
    epidemiológico (inclusive anos com 53 semanas), os limiares de risco, a limpeza dos dados, a ausência de
    informação do futuro nas variáveis de entrada, a divisão temporal, o desempenho mínimo dos modelos em relação
    à linha de base, cenários de alerta (epidêmico → alto; calmo → baixo), a recusa de entradas inválidas e as
    regras do banco de dados:</p>
    <pre class='saida'>{esc(testes_txt)}</pre>
    """)

    # ------------------------------------------------------------- IMPACTO
    H.append(f"""
    <h1>4. Impacto e Objetivos de Desenvolvimento Sustentável</h1>
    <p>O Sentinela contribui com o <b>ODS 3</b> ao antecipar a resposta a uma doença transmissível (meta 3.3) e ao
    fortalecer a capacidade de alerta precoce do município (meta 3.d). Contribui com o <b>ODS 11</b> ao usar dados
    abertos para orientar ações públicas nos bairros e tornar a cidade mais preparada para epidemias sazonais.</p>
    <h2>4.1 Como medir o impacto</h2>
    {tabela_html(pd.DataFrame([
        ["Inícios de risco alto avisados com 2 semanas", "Proporção de temporadas em que o alerta veio antes do início", "Teste histórico (Tabela de antecipação)"],
        ["Alarmes falsos por temporada", "Semanas previstas como alto que não foram", "Teste histórico"],
        ["Erro médio da previsão de casos (MAE)", "Comparado com a linha de base", "Teste histórico"],
        ["Tempo de resposta da prefeitura", "Dias entre o alerta e o início das ações (mutirão, reforço de equipe)", "Uso real (piloto)"],
        ["Internações e óbitos por dengue por 100 mil hab.", "Comparar temporadas antes e depois da adoção", "Uso real (DataSUS)"],
        ["Alcance dos avisos à população", "Pessoas alcançadas por SMS/WhatsApp/redes", "Uso real"],
    ], columns=["Indicador", "Como calcular", "Quando"]), "Indicadores de impacto.", next(nt))}
    """)

    # ----------------------------------------------------------- LIMITAÇÕES
    H.append(f"""
    <h1>5. Limitações e próximos passos</h1>
    <h2>5.1 Limitações do protótipo</h2>
    <ul>
      {'<li><b>Os resultados apresentados vêm da base de demonstração simulada</b>; os números com dados reais podem ser diferentes.</li>' if simulado else ''}
      <li>Os limiares de risco (10 e 50 casos por 100 mil por semana) são um critério do grupo e devem ser
      calibrados com a vigilância local.</li>
      <li>O período de teste tem poucas temporadas epidêmicas, então as métricas de antecipação se baseiam em
      poucos eventos.</li>
      <li>Atraso de notificação: as últimas semanas da série real chegam incompletas, o que pode subestimar o risco
      atual.</li>
      <li>O clima é um valor por município e não há detalhamento por bairro.</li>
      <li>O modelo aprende padrões do passado; mudanças de sorotipo, de vacinação ou de política de testagem podem
      alterar esses padrões.</li>
      <li>O sistema não substitui profissionais de saúde nem dá orientação médica individual.</li>
    </ul>
    <h2>5.2 Próximos passos</h2>
    <ul>
      <li>Calibrar os limiares com a vigilância municipal e testar em mais cidades (Itaúna já está configurada).</li>
      <li>Incluir chuva acumulada e dados de levantamento de infestação (LIRAa).</li>
      <li>Coletar dados de ovitrampas por bairro com sensores e ESP32 e gravar no mesmo banco.</li>
      <li>Enviar o alerta automaticamente por WhatsApp/e-mail e publicar um painel para a população.</li>
    </ul>""")

    # ------------------------------------------------------------- CONCLUSÃO
    H.append(f"""
    <h1>6. Conclusões</h1>
    <p>O Sentinela Dengue mostra que dados abertos, já publicados toda semana, podem virar um alerta prático com
    técnicas estudadas até a Fase 3: programação em Python, tratamento e análise estatística com Pandas, gráficos,
    classificação e regressão com scikit-learn e banco de dados relacional. A POC funciona de ponta a ponta, da
    coleta ao alerta gravado no banco, e é verificada por {n_testes} testes automatizados.</p>
    <p>Um aprendizado importante foi comparar sempre com uma linha de base simples. Em acurácia geral, repetir o
    risco atual já é difícil de superar. O ganho do modelo está em <b>prever as mudanças de nível</b>, que é o
    que dá à prefeitura tempo para agir. A ferramenta foi pensada para apoiar a vigilância em saúde, com
    transparência sobre as limitações, e pode ser ampliada para outros municípios e outras arboviroses
    (zika e chikungunya) com a mesma API.</p>

    <h1>7. Links e referências</h1>
    <table class='tab links'>
      <tr><th>Item</th><th>Link</th></tr>
      <tr><td>Vídeo de demonstração (YouTube, não listado)</td><td>{esc(LINK_VIDEO)}</td></tr>
      <tr><td>Repositório GitHub (privado, compartilhado com <code>fiap-tutoria</code>)</td><td>{esc(LINK_REPO)}</td></tr>
    </table>
    <h2>Instruções de execução</h2>
    <pre class='code'>pip install -r requirements.txt
python src/coleta_infodengue.py       # dados reais (ou: python src/gerar_base_demo.py)
python src/main.py --pipeline         # fluxo completo
python src/main.py                    # menu interativo
python -m pytest -v                   # testes</pre>
    <h2>Referências</h2>
    <ul>
      <li>InfoDengue – FIOCRUZ / FGV. API de dados por município: https://info.dengue.mat.br/services/api</li>
      <li>Organização das Nações Unidas. Objetivos de Desenvolvimento Sustentável (ODS 3 e ODS 11): https://brasil.un.org/pt-br/sdgs</li>
      <li>Documentação do scikit-learn: https://scikit-learn.org</li>
      <li>Documentação do pandas: https://pandas.pydata.org</li>
    </ul>""")

    return f"""<!DOCTYPE html><html lang='pt-BR'><head><meta charset='utf-8'>
    <title>Sentinela Dengue - Global Solution 2026.2</title><style>{css}</style></head>
    <body>{''.join(H)}</body></html>"""


def main() -> int:
    saida = config.RAIZ / "docs" / "relatorio_entrega.html"
    saida.write_text(montar(), encoding="utf-8")
    print(f"Documento gerado: {saida}")
    print("Abra no navegador e use Imprimir (Ctrl+P) -> Salvar como PDF.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
