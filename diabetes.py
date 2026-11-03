import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, VotingClassifier)

arq_train = glob.glob("kaggle-training set/**/*.csv", recursive=True)[0]
arq_test = glob.glob("kaggle-test/**/*.csv", recursive=True)[0]
print("Treino:", arq_train, "| Teste:", arq_test)
train = pd.read_csv(arq_train)
test = pd.read_csv(arq_test)

y = train["class"]
ids = test["id"]
X = train.drop(columns=["class"])
X_test = test.copy()

def converte_numericas(df):
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            if num.notna().sum() >= 0.8 * df[c].notna().sum():
                df[c] = num
    return df

X = converte_numericas(X)
X_test = converte_numericas(X_test)

def padroniza(df, col, regras):
    if col in df:
        v = df[col].astype(str).str.strip().str.lower()
        novo = v.map(lambda x: next((k for k, lst in regras.items() if x in lst), "outro"))
        df[col] = novo.where(df[col].notna(), np.nan)

def limpa(df):
    padroniza(df, "GENDER", {"F": ["female", "f"], "M": ["male", "m"]})
    padroniza(df, "SMOKER", {"never": ["never", "n", "no"], "former": ["former", "ex-smoker"],
                             "current": ["current", "yes"]})
    padroniza(df, "FAMILY_HISTORY", {"no": ["no", "n", "0", "não", "nao"],
                                     "yes": ["yes", "y", "1", "sim"], "unknown": ["unknown"]})
    if "HEIGHT" in df:
        h = df["HEIGHT"]; h = h.where(h >= 3, h * 100)
        df["HEIGHT"] = h.where(h.between(100, 230))
    if "AGE" in df:
        a = df["AGE"]
        if "BIRTH_YEAR" in df and "EXAM_DATE" in df:
            ano = pd.to_datetime(df["EXAM_DATE"], errors="coerce").dt.year
            nasc = pd.to_numeric(df["BIRTH_YEAR"], errors="coerce")
            a = a.where(a.between(0, 110), ano - nasc)
        df["AGE"] = a.where(a.between(0, 110))
    if "WEIGHT" in df:
        df["WEIGHT"] = df["WEIGHT"].where(df["WEIGHT"].between(20, 300))
    if "HEMOGLOBIN" in df:
        hb = df["HEMOGLOBIN"]; hb = hb.where(hb <= 20, hb / 10.929 + 2.15)
        df["HEMOGLOBIN"] = hb.where(hb.between(3, 20))
    if "GLUCOSE" in df:
        g = df["GLUCOSE"]; g = g.where(g >= 30, g * 18.016)
        df["GLUCOSE"] = g.where(g.between(30, 700))
    if "TRIGLYCERIDES" in df:
        t = df["TRIGLYCERIDES"]; t = t.where(t >= 20, t * 88.57)
        df["TRIGLYCERIDES"] = t.where(t.between(20, 3000))
    if "SODIUM" in df:
        df["SODIUM"] = df["SODIUM"].where(df["SODIUM"].between(100, 180))
    return df

X = limpa(X)
X_test = limpa(X_test)
SOBRA = ["id", "PATIENT_CODE", "EXAM_DATE", "BIRTH_YEAR"]
X = X.drop(columns=SOBRA, errors="ignore")
X_test = X_test.drop(columns=SOBRA, errors="ignore")

for df in (X, X_test):
    if "BLOOD_PRESSURE" in df and not pd.api.types.is_numeric_dtype(df["BLOOD_PRESSURE"]):
        pa = df["BLOOD_PRESSURE"].astype(str).str.split("/", expand=True)
        if pa.shape[1] == 2:
            df["PA_SIS"] = pd.to_numeric(pa[0], errors="coerce")
            df["PA_DIA"] = pd.to_numeric(pa[1], errors="coerce")
            df.drop(columns=["BLOOD_PRESSURE"], inplace=True)
    if "HEIGHT" in df and "WEIGHT" in df:
        df["IMC"] = df["WEIGHT"] / ((df["HEIGHT"] / 100) ** 2)
    if "GLUCOSE" in df and "HEMOGLOBIN" in df:
        df["GLIC_HEMO"] = df["GLUCOSE"] / df["HEMOGLOBIN"]
    if "GLUCOSE" in df and "IMC" in df:
        df["GLIC_IMC"] = df["GLUCOSE"] * df["IMC"]
    if "AGE" in df and "IMC" in df:
        df["IDADE_IMC"] = df["AGE"] * df["IMC"]
    if "TRIGLYCERIDES" in df and "GLUCOSE" in df:
        df["TG_GLIC"] = df["TRIGLYCERIDES"] / df["GLUCOSE"]
    if "PA_SIS" in df and "PA_DIA" in df:
        df["PULSO"] = df["PA_SIS"] - df["PA_DIA"]
        
    # Feature Engineering Manual (Risk Score Clínico)
    risk = np.zeros(len(df))
    if "GLUCOSE" in df: risk += (df["GLUCOSE"] > 100).astype(int)
    if "IMC" in df: risk += (df["IMC"] > 30).astype(int)
    if "PA_SIS" in df: risk += (df["PA_SIS"] > 130).astype(int)
    if "AGE" in df: risk += (df["AGE"] > 50).astype(int)
    df["RISK_SCORE"] = risk

n = len(X)
todos = pd.get_dummies(pd.concat([X, X_test]), dtype=float)
colunas = list(todos.columns)
X, X_test = todos.iloc[:n].copy(), todos.iloc[n:].copy()

print("\n--- A ABORDAGEM BRUTALISTA ---")
# 1. SEM StandardScaler. Árvores cagam para outliers.
# 2. Imputação com -999. Separa os nulos num buraco isolado que a árvore acha fácil.

pipe_rf = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-999, add_indicator=True)),
    ('clf', RandomForestClassifier(n_estimators=600, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1))
])

pipe_et = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-999, add_indicator=True)),
    ('clf', ExtraTreesClassifier(n_estimators=600, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1))
])

pipe_hgb = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-999, add_indicator=True)),
    ('clf', HistGradientBoostingClassifier(learning_rate=0.03, max_iter=500, max_depth=5, l2_regularization=0.1, class_weight='balanced', random_state=42))
])

# Soft Voting: Apenas tira a média das probabilidades. Sem meta-modelo para decorar o treino.
voting_clf = VotingClassifier(
    estimators=[('rf', pipe_rf), ('et', pipe_et), ('hgb', pipe_hgb)],
    voting='soft',
    weights=[1, 1, 1.2] # Damos um peso levemente maior ao HistGradientBoosting
)

cv = StratifiedKFold(5, shuffle=True, random_state=42)

print("Calculando probabilidades via validação cruzada para otimização do limiar...")
probs_cv = cross_val_predict(voting_clf, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

melhor_f1 = 0
melhor_limiar = 0.5
for limiar in np.arange(0.20, 0.81, 0.01):
    preds = (probs_cv >= limiar).astype(int)
    score = f1_score(y, preds)
    if score > melhor_f1:
        melhor_f1 = score
        melhor_limiar = limiar

print(f"Voting F1 Bruto (CV) = {melhor_f1:.3f} | Melhor Limiar = {melhor_limiar:.2f}\n")

# Treina com todos os dados e gera o CSV
voting_clf.fit(X, y)
probs_teste = voting_clf.predict_proba(X_test)[:, 1]
preds_finais = (probs_teste >= melhor_limiar).astype(int)

pd.DataFrame({"id": ids, "class": preds_finais}).to_csv("submission_bruta.csv", index=False)
print("Ficheiro submission_bruta.csv gerado com sucesso!")
