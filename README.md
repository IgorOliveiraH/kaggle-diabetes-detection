# 🩺 Detecção de Diabetes - Kaggle Competition

Este repositório contém a solução desenvolvida para a competição de Mineração de Dados (UFU) no Kaggle, focada na classificação preditiva de diabetes utilizando dados tabulares clínicos.

### 🎯 Objetivo
Construir e otimizar um modelo de Machine Learning capaz de classificar pacientes em grupos de risco para diabetes, auxiliando no diagnóstico médico com base em variáveis de saúde.

### 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Bibliotecas Principais:** Pandas, NumPy, Scikit-Learn
* **Técnicas de ML:** Limpeza de dados, Feature Engineering e Voting/Stacking Ensemble.

### 🧠 Abordagem e Metodologia
A solução foi estruturada através de um pipeline completo de mineração de dados. Para maximizar a capacidade de generalização e evitar *overfitting*, a arquitetura final do modelo implementa uma estratégia de **Voting/Stacking Ensemble**. Essa abordagem combinou as predições de múltiplos algoritmos base para entregar um classificador mais robusto e estável.

### 🏆 Resultados
* **Métrica de Avaliação:** F1-Score
* **Score no Leaderboard:** 0.89904
* O uso de técnicas avançadas de *Ensemble* garantiu uma classificação altamente competitiva na base de testes oculta.

### 🚀 Como Executar Localmente

1. Clone este repositório:
```bash
git clone [https://github.com/IgorOliveiraH/kaggle-diabetes-detection.git](https://github.com/IgorOliveiraH/kaggle-diabetes-detection.git)
