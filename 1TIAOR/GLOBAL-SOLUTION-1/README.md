# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href="https://www.fiap.com.br/">
  <img src="../../assets/logo-fiap.png" 
       alt="FIAP - Faculdade de Informática e Administração Paulista" 
       width="40%">
</a>
</p>

<br>

# Sentinela Dengue: alerta antecipado de risco de dengue com IA e dados abertos

## Grupo Sentinela

## 👨‍🎓 Integrantes: 
- <a href="https://www.linkedin.com/in/raiane-sirlene-silva/">Raiane Sirlene da Silva (RM575984)</a>
- <a href="LINK-DO-PERFIL">Rodrigo Textor de Textor (RM574243)</a>
- <a href="LINK-DO-PERFIL">Pedro Victor Pires (RM575181)</a>
  
### Coordenador(a)
- <a href="https://www.linkedin.com/in/andregodoichiovato/">André Godoi Chiovato</a>

## 📜 Descrição

**Global Solution 2026.2 · Desafio CitizenTech · Turma 1TIAOR 2026/2**

### O problema
A dengue é a arbovirose de maior impacto no Brasil. Quando a vigilância percebe a alta de casos, as UBS e UPAs normalmente já estão cheias. Grande parte da resposta (mutirões de limpeza, visitas casa a casa, reforço de equipes, compra de insumos e alerta à população) leva dias para ser organizada, por isso **saber com antecedência** faz diferença.

### Quem é beneficiado
- **Equipes de vigilância epidemiológica e de saúde** do município, que recebem um alerta de risco com 2 semanas de antecedência e uma lista de ações sugeridas.
- **Moradores**, especialmente idosos, crianças e pessoas que dependem do SUS, que podem ser avisados antes do pico.

### A solução
O Sentinela Dengue é uma POC em Python que:
1. **Coleta** a série semanal de dengue do município na API pública do **InfoDengue** (FIOCRUZ/FGV), com casos, temperatura, umidade e população;
2. **Trata** os dados: remove duplicatas, ordena as semanas, descarta registros inválidos, interpola clima faltante e calcula a incidência por 100 mil habitantes;
3. **Analisa** com estatística descritiva, sazonalidade, incidência anual e correlação de Spearman entre clima defasado e casos (Pandas, NumPy, Matplotlib, Seaborn);
4. **Prevê** com Machine Learning (scikit-learn):
   - **classificação** do risco daqui a 2 semanas (baixo / médio / alto), comparando Regressão Logística, Árvore de Decisão e Random Forest;
   - **regressão** do número de casos daqui a 2 semanas (Regressão Linear sobre log dos casos);
   - todos os modelos são comparados com uma **linha de base ingênua** (persistência);
5. **Armazena** a série e o histórico de alertas em um banco **SQLite** com chaves, restrições e consultas SQL;
6. **Emite o alerta** no terminal, com risco, probabilidade, casos estimados e ações recomendadas, e permite simular cenários "e se...?".

### Classes de risco (critério do projeto)
| Risco | Incidência semanal (casos por 100 mil hab.) |
|---|---|
| baixo | menor que 10 |
| médio | de 10 a menos de 50 |
| alto | 50 ou mais |

### ODS relacionados
- **ODS 3 – Saúde e bem-estar** (meta 3.3: combater doenças transmissíveis; meta 3.d: alerta precoce e gestão de riscos à saúde).
- **ODS 11 – Cidades e comunidades sustentáveis**: cidades mais seguras e resilientes, com uso de dados abertos para orientar a ação do poder público nos bairros.

### Como medir o impacto
- Proporção de inícios de período de risco alto avisados com 2 semanas de antecedência;
- Número de alarmes falsos por temporada;
- Erro médio da previsão de casos (MAE) em relação à linha de base;
- Em uso real: internações e óbitos por dengue por 100 mil habitantes e tempo de resposta da prefeitura, antes e depois da adoção.

### ⚠️ Sobre os dados
- **Dados reais:** API do InfoDengue (`src/coleta_infodengue.py`). O município padrão é Belo Horizonte (IBGE 3106200); Itaúna (3133808) também está configurada.
- **Base de demonstração SIMULADA:** `data/demo/infodengue_demo_SIMULADO.csv`, gerada por `src/gerar_base_demo.py` com o mesmo formato da API. Ela **não representa nenhum município real** e serve para rodar o projeto sem internet e para os testes automatizados. Todo arquivo produzido a partir dela recebe o sufixo `_SIMULADO`.
- Quando existe a base real em `data/raw/`, o programa a usa automaticamente.

## 📁 Estrutura de pastas 

* **`data/`** Bases de dados: `raw/` (dados reais baixados do InfoDengue), `demo/` (base de demonstração simulada), `processed/` (dados tratados) e o banco `sentinela_dengue.db`.
* **`docs/`** Documentação: `figuras/` (gráficos gerados), `resultados/` (tabelas de métricas e saídas de execução) e os documentos do projeto.
* **`src/`** Código-fonte:
  * `config.py` — parâmetros (município, limiares, horizonte, caminhos);
  * `coleta_infodengue.py` — coleta dos dados reais;
  * `gerar_base_demo.py` — gera a base de demonstração simulada;
  * `semana_epi.py` — calendário de semanas epidemiológicas;
  * `tratamento.py` — limpeza e engenharia de atributos;
  * `analise.py` — estatística descritiva e gráficos;
  * `modelo.py` — classificação, regressão, métricas e gráficos de resultado;
  * `banco.py` — banco SQLite e consultas SQL;
  * `alerta.py` — geração do alerta e simulação de cenários;
  * `main.py` — programa principal (menu e fluxo completo);
  * `gerar_relatorio.py` — gera o documento de entrega (HTML → PDF) com os resultados atualizados.
* **`tests/`** Testes automatizados (pytest).
* **`README.md`** Este arquivo.

## 📎 Links e Observações

### 🔗 Links do Projeto

* **[API InfoDengue](https://info.dengue.mat.br/services/api)** - Fonte dos dados reais.

### 🧠 Decisões Técnicas

> **Dados do InfoDengue:** fonte aberta, oficial e semanal, que já traz o clima do município, sem necessidade de cadastro.
>
> **Previsão de 2 semanas:** prazo mínimo para a prefeitura organizar mutirões, escalas e insumos. Horizontes maiores perdem precisão.
>
> **Divisão temporal treino/teste** (treino até 2022, teste de 2023 em diante): simula o uso real e impede que o modelo "veja o futuro". Uma divisão aleatória daria métricas otimistas e enganosas.
>
> **Clima defasado (1 a 4 semanas):** o ciclo do *Aedes aegypti* (ovo, larva e mosquito adulto) mais o período de incubação fazem o clima de hoje aparecer nos casos semanas depois.
>
> **Escala logarítmica dos casos:** a série tem semanas com 10 e com 10 mil casos; o log evita que os picos dominem o ajuste.
>
> **Escolha do classificador pelo recall da classe "alto":** deixar passar o início de uma epidemia custa mais caro que um alarme falso. O empate é decidido pelo F1 macro.
>
> **SQLite:** banco relacional em um único arquivo, sem servidor, com SQL padrão (fácil de migrar para Oracle/PostgreSQL).
>
> **Linha de base (persistência):** todo modelo é comparado com "daqui a 2 semanas será igual a hoje". Só faz sentido usar ML se ele ganhar dessa regra simples.

### 📢 Observações Gerais

* **Participação na Competição:** [x] Sim, aceitamos participar / [ ] Não vamos participar.
* O sistema é uma ferramenta de **apoio à decisão** e não substitui a avaliação da equipe de vigilância em saúde.

## 🔧 Como executar o código

**Pré-requisitos:** Python 3.10 ou superior (testado com 3.13), terminal e internet (apenas para baixar os dados reais).

```bash
# 1) Clonar o repositório e entrar na pasta do projeto
git clone  https://github.com/Silva-Raiane/global_solutions-1_FIAP.git
cd FIAP-PORTFOLIO-AI/1TIAOR/GLOBAL-SOLUTION-1

# 2) (Opcional) criar ambiente virtual
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/Mac: source .venv/bin/activate

# 3) Instalar as bibliotecas
pip install -r requirements.txt

# 4) Baixar os dados REAIS do InfoDengue (Belo Horizonte)
python src/coleta_infodengue.py
#    Itaúna: python src/coleta_infodengue.py --geocode 3133808
#    Sem internet: python src/gerar_base_demo.py  (base SIMULADA)

# 5) Rodar o fluxo completo (tratamento, análise, modelos e alerta)
python src/main.py --pipeline

# 6) Usar o menu interativo (simular cenários e consultar o banco)
python src/main.py

# 7) Rodar os testes automatizados
python -m pytest -v

# 8) Gerar o documento de entrega (abrir o HTML no navegador e salvar como PDF)
python src/gerar_relatorio.py
```

Os gráficos são salvos em `docs/figuras/`, as métricas em `docs/resultados/` e o banco em `data/sentinela_dengue.db`.

---

## 📋 Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/fiap-tutoria/FIAP-PORTFOLIO-AI">TEMPLATE - FIAP PORTFÓLIO AI</a> por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">FIAP</a> está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>
