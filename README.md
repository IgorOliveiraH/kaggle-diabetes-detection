# Detecção de Diabetes — Mineração de Dados

**Integrantes:** Igor Oliveira, Leandro Marques e Adiel Pereira

Classificação de pacientes em **controle (0)** ou **diabetes (1)** a partir de dados tabulares clínicos, na competição Kaggle da disciplina. A métrica é o F1. O conjunto de teste não é usado em nenhuma etapa de treino ou validação: ele só recebe a predição final.

## Conteúdo da pasta

```
deteccao-de-diabetes/
├── kaggle-training set/     dados de treino (CSV com a coluna "class")
├── kaggle-test/             dados de teste (CSV sem a coluna "class")
├── diabetes_final.py        modelo final (gera a submissão principal)
├── diabetes_hardmargin.py   experimento com casos de fronteira (pesos focais, KNN e limiar)
├── requirements.txt         bibliotecas Python
└── README.md                este arquivo
```

## Requisitos

- Linux; os passos abaixo são para Fedora (40 ou mais recente)
- Python 3.10 ou mais recente
- Bibliotecas: numpy, pandas e scikit-learn (versões em `requirements.txt`)
- Sem GPU; cerca de 1 GB de RAM livre

Testado com Python 3.13 em dois ambientes: scikit-learn 1.5.2 + pandas 2.2.3 e scikit-learn 1.9.1 + pandas 3.0.5.

## Instalação (Fedora)

```bash
sudo dnf install -y python3 python3-pip
cd deteccao-de-diabetes
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

O ambiente virtual (`.venv`) é opcional, mas evita conflito com os pacotes do sistema.

## Configuração

Nenhuma configuração extra é necessária.

- Os dados ficam nas pastas `kaggle-training set/` e `kaggle-test/`. Cada script usa o primeiro arquivo `.csv` que encontrar em cada pasta. O nome da pasta de treino tem um espaço, então use aspas no terminal: `ls "kaggle-training set"`.
- Os parâmetros principais estão no topo de `diabetes_final.py`: `LIMIAR` (0.55), `PESOS` (peso 1,4 na classe 1) e `SEED` (42).

## Compilação

Não há etapa de compilação: Python é interpretado. Para conferir só a sintaxe:

```bash
python3 -m py_compile diabetes_final.py diabetes_hardmargin.py
```

## Execução

Rode os comandos **dentro da pasta do projeto**:

```bash
cd deteccao-de-diabetes
source .venv/bin/activate
python3 diabetes_final.py
```

A execução leva alguns minutos (cerca de 5 num notebook comum) e gera:

| Arquivo | Conteúdo |
| --- | --- |
| `submission.csv` | predições do teste (`id`, `class`), no formato do Kaggle |
| `resultados_cv.csv` | AUC e F1 de todos os modelos na validação cruzada |

Experimento opcional, com os casos de fronteira:

```bash
python3 diabetes_hardmargin.py
```

Ele gera `submission_hardmargin.csv` e `resultados_hardmargin.csv`.

## O que o `diabetes_final.py` faz

1. **Limpeza:** padroniza categorias escritas de formas diferentes, converte unidades (altura em metros, glicose e triglicerídeos em mmol/L, HbA1c em mmol/mol), corrige peso em libras e recalcula idades inválidas pela data do exame.
2. **Features:** pressão sistólica e diastólica, IMC e um escore de risco (glicose > 100, IMC > 30, pressão > 130).
3. **Validação:** 5-fold estratificado repetido 3 vezes, só no treino (previsões out-of-fold). Pacientes repetidos ficam sempre no mesmo fold.
4. **Modelos:** KNN, Árvore de Decisão e MLP como ponto de partida, e um Voting de Random Forest, ExtraTrees e HistGradientBoosting.
5. **Casos de fronteira:** filtro de ruído nos rótulos do treino (Brodley & Friedl, 1999) e critério clínico da ADA quando o modelo fica em dúvida. O script só adota cada técnica se ela ganhar na validação.

## Resultados

| Script | F1 na validação (OOF) | Kaggle público |
| --- | --- | --- |
| `diabetes_final.py` | 0,9117 | 0,899 |
| `diabetes_hardmargin.py` | 0,9110 | 0,90822 |

Baseline da disciplina no Kaggle: 0,87.

## Reprodutibilidade

`SEED = 42` fixa todas as sementes aleatórias. Rodar de novo, com as mesmas versões das bibliotecas, gera os mesmos arquivos.

## Problemas comuns

| Erro | Causa e solução |
| --- | --- |
| `IndexError: list index out of range` | O script foi rodado fora da pasta do projeto, ou falta o CSV em `kaggle-training set/` ou `kaggle-test/`. |
| `ModuleNotFoundError: No module named 'sklearn'` | O ambiente virtual não está ativo: rode `source .venv/bin/activate`. |
