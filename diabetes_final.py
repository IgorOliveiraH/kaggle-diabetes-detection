#!/usr/bin/env python3
"""
Detecção de Diabetes - Mineração de Dados
Igor Oliveira, Leandro Marques e Adiel Pereira

Uso (na pasta com "kaggle-training set" e "kaggle-test"): python3 diabetes_final.py
Saídas: submission.csv (submissão do Kaggle) e resultados_cv.csv (validação dos modelos)
"""
import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

# limiar de decisão da classe 1, definido na validação
LIMIAR = 0.55
PESOS = {0: 1.0, 1: 1.4}
SEED = 42
DESCARTAR = ["id", "PATIENT_CODE", "EXAM_DATE", "BIRTH_YEAR"]

arq_train = glob.glob("kaggle-training set/**/*.csv", recursive=True)[0]
arq_test = glob.glob("kaggle-test/**/*.csv", recursive=True)[0]
train = pd.read_csv(arq_train)
test = pd.read_csv(arq_test)

y = train["class"].astype(int)
ids = test["id"]
X = train.drop(columns=["class"])
X_test = test.copy()


# coluna vira numérica só se >= 80% dos valores do treino forem números
def colunas_numericas(df):
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
            continue
        num = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        if num.notna().sum() >= 0.8 * df[c].notna().sum():
            cols.append(c)
    return cols


def converte_numericas(df, cols):
    for c in cols:
        if c in df and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return df


cols_num = colunas_numericas(X)
X = converte_numericas(X, cols_num)
X_test = converte_numericas(X_test, cols_num)


def padroniza(df, col, regras):
    if col in df:
        v = df[col].astype(str).str.strip().str.lower()
        novo = v.map(lambda x: next((k for k, lst in regras.items() if x in lst), "outro"))
        df[col] = novo.where(df[col].notna(), np.nan)


# padroniza categorias e unidades: altura m -> cm, HbA1c mmol/mol -> %, glicose e triglicerídeos mmol/L -> mg/dL
def limpa(df):
    padroniza(df, "GENDER", {"F": ["female", "f"], "M": ["male", "m"]})
    padroniza(df, "SMOKER", {"never": ["never", "n", "no"], "former": ["former", "ex-smoker"], "current": ["current", "yes"]})
    padroniza(df, "FAMILY_HISTORY", {"no": ["no", "n", "0", "não", "nao"], "yes": ["yes", "y", "1", "sim"], "unknown": ["unknown"]})
    if "HEIGHT" in df:
        h = df["HEIGHT"]; h = h.where(h >= 3, h * 100); df["HEIGHT"] = h.where(h.between(100, 230))
    if "AGE" in df:
        a = df["AGE"]
        if "BIRTH_YEAR" in df and "EXAM_DATE" in df:
            ano = pd.to_datetime(df["EXAM_DATE"], errors="coerce").dt.year
            nasc = pd.to_numeric(df["BIRTH_YEAR"], errors="coerce")
            a = a.where(a.between(0, 110), ano - nasc)
        df["AGE"] = a.where(a.between(0, 110))
    if "WEIGHT" in df: df["WEIGHT"] = df["WEIGHT"].where(df["WEIGHT"].between(20, 300))
    if "HEMOGLOBIN" in df: hb = df["HEMOGLOBIN"]; hb = hb.where(hb <= 20, hb / 10.929 + 2.15); df["HEMOGLOBIN"] = hb.where(hb.between(3, 20))
    if "GLUCOSE" in df: g = df["GLUCOSE"]; g = g.where(g >= 30, g * 18.016); df["GLUCOSE"] = g.where(g.between(30, 700))
    if "TRIGLYCERIDES" in df: t = df["TRIGLYCERIDES"]; t = t.where(t >= 20, t * 88.57); df["TRIGLYCERIDES"] = t.where(t.between(20, 3000))
    return df


X = limpa(X)
X_test = limpa(X_test)

# features: pressão sistólica/diastólica, IMC e escore de risco
for df in (X, X_test):
    if "BLOOD_PRESSURE" in df and not pd.api.types.is_numeric_dtype(df["BLOOD_PRESSURE"]):
        pa = df["BLOOD_PRESSURE"].astype(str).str.split("/", expand=True)
        if pa.shape[1] == 2:
            df["PA_SIS"] = pd.to_numeric(pa[0], errors="coerce")
            df["PA_DIA"] = pd.to_numeric(pa[1], errors="coerce")
            df.drop(columns=["BLOOD_PRESSURE"], inplace=True)
    if "HEIGHT" in df and "WEIGHT" in df: df["IMC"] = df["WEIGHT"] / ((df["HEIGHT"] / 100) ** 2)
    risk = np.zeros(len(df))
    if "GLUCOSE" in df: risk += (df["GLUCOSE"] > 100).to_numpy(dtype=int)
    if "IMC" in df: risk += (df["IMC"] > 30).to_numpy(dtype=int)
    if "PA_SIS" in df: risk += (df["PA_SIS"] > 130).to_numpy(dtype=int)
    df["RISK_SCORE"] = risk


# mesma categoria escrita de formas diferentes (Light, light, LIGHT) vira uma só
def normaliza_categorias(df):
    df = df.copy()
    for c in ["PHYSICAL_ACTIVITY", "BLOOD_TYPE", "COLLECTION_SITE", "LAB_DEVICE", "STUDY_PROTOCOL"]:
        if c in df and not pd.api.types.is_numeric_dtype(df[c]):
            v = df[c].astype(str).str.strip().str.lower().str.replace(r"[\s_\-]+", " ", regex=True)
            v = v.replace({"sedentry": "sedentary"})
            df[c] = v.where(df[c].notna(), np.nan)
    return df


# peso em libras (IMC impossível) e idade inválida, recalculada pela data do exame
def corrige_valores(df):
    df = df.copy()
    if "WEIGHT" in df and "HEIGHT" in df:
        h2 = (df["HEIGHT"] / 100) ** 2
        lb = ((df["WEIGHT"] / h2) > 55) & ((df["WEIGHT"] / 2.20462 / h2).between(15, 50))
        df.loc[lb, "WEIGHT"] = df.loc[lb, "WEIGHT"] / 2.20462
        df["IMC"] = df["WEIGHT"] / h2
    if "AGE" in df:
        a = df["AGE"].where(df["AGE"].between(1, 110))
        if "BIRTH_YEAR" in df and "EXAM_DATE" in df:
            ano = pd.to_numeric(df["EXAM_DATE"].astype(str).str.extract(r"((?:19|20)\d{2})", expand=False), errors="coerce")
            calc = ano - pd.to_numeric(df["BIRTH_YEAR"], errors="coerce")
            a = a.fillna(calc.where(calc.between(1, 110)))
        df["AGE"] = a
    risk = np.zeros(len(df))
    if "GLUCOSE" in df: risk += (df["GLUCOSE"] > 100).to_numpy(dtype=int)
    if "IMC" in df: risk += (df["IMC"] > 30).to_numpy(dtype=int)
    if "PA_SIS" in df: risk += (df["PA_SIS"] > 130).to_numpy(dtype=int)
    df["RISK_SCORE"] = risk
    return df


# dummies só com as categorias do treino; o teste é alinhado a elas
def matrizes(Xa, Xb):
    tr = Xa.drop(columns=DESCARTAR, errors="ignore")
    te = Xb.drop(columns=DESCARTAR, errors="ignore")
    numericas = [c for c in tr.columns if pd.api.types.is_numeric_dtype(tr[c])]
    tr = pd.get_dummies(tr, dtype=float)
    te = pd.get_dummies(te, dtype=float).reindex(columns=tr.columns)
    dummies = [c for c in tr.columns if c not in numericas]
    te[dummies] = te[dummies].fillna(0.0)
    for c in numericas:
        te[c] = pd.to_numeric(te[c], errors="coerce")
    return tr, te, numericas


# matriz original (pipeline do diabetes.py) e matriz corrigida
X_treino, X_teste, _ = matrizes(X, X_test)
X_treino_c, X_teste_c, numericas = matrizes(corrige_valores(normaliza_categorias(X)),
                                           corrige_valores(normaliza_categorias(X_test)))

# modelos do ensemble
pipe_rf = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
                    ('clf', RandomForestClassifier(n_estimators=600, max_depth=12, class_weight=PESOS, random_state=SEED, n_jobs=-1))])
pipe_et = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
                    ('clf', ExtraTreesClassifier(n_estimators=600, max_depth=12, class_weight=PESOS, random_state=SEED, n_jobs=-1))])
pipe_hgb = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
                     ('clf', HistGradientBoostingClassifier(learning_rate=0.03, max_iter=500, max_depth=5, min_samples_leaf=10, l2_regularization=0.1, class_weight='balanced', random_state=SEED))])
voting_clf = VotingClassifier(estimators=[('rf', pipe_rf), ('et', pipe_et), ('hgb', pipe_hgb)], voting='soft', weights=[1.2, 1.2, 1])

# ExtraTrees com mediana: com -999 o corte sorteado quase sempre cai entre -999 e os valores reais
pipe_et_c = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                      ('clf', ExtraTreesClassifier(n_estimators=600, max_depth=12, class_weight=PESOS, random_state=SEED, n_jobs=-1))])
voting_c = VotingClassifier(estimators=[('rf', clone(pipe_rf)), ('et', pipe_et_c), ('hgb', clone(pipe_hgb))], voting='soft', weights=[1.2, 1.2, 1])


# KNN e MLP: escala só as colunas numéricas (dummies ficam 0/1)
def escalado(clf):
    num = Pipeline([('imputer', SimpleImputer(strategy='median')),
                    ('scaler', QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=SEED))])
    prep = ColumnTransformer([('num', num, numericas),
                              ('faltante', MissingIndicator(error_on_new=False), numericas)],
                             remainder='passthrough')
    return Pipeline([('prep', prep), ('clf', clf)])


# (nome, modelo, usa matriz corrigida, limiar)
modelos = [
    ("KNN", escalado(KNeighborsClassifier(n_neighbors=15, weights='distance')), True, 0.5),
    ("Arvore de Decisao", Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
                                    ('clf', DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight=PESOS, random_state=SEED))]), True, 0.5),
    ("MLP", escalado(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=1000, early_stopping=True, random_state=SEED)), True, 0.5),
    ("Random Forest", pipe_rf, False, 0.5),
    ("Extra Trees (-999)", pipe_et, False, 0.5),
    ("Extra Trees (mediana)", pipe_et_c, True, 0.5),
    ("HistGradientBoosting", pipe_hgb, False, 0.5),
    ("Voting original", voting_clf, False, LIMIAR),
    ("Voting corrigido", voting_c, True, LIMIAR),
]

# 5-fold estratificado; paciente repetido (PATIENT_CODE) fica sempre no mesmo fold
def gera_folds(seed):
    if "PATIENT_CODE" in train and train["PATIENT_CODE"].dropna().duplicated().any():
        grupos = train["PATIENT_CODE"].fillna(pd.Series([f"sem_{i}" for i in range(len(train))], index=train.index))
        return list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X_treino, y, grupos))
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_treino, y))


def oof(modelo, M, seed=SEED):
    p = np.zeros(len(y))
    for tr, va in gera_folds(seed):
        m = clone(modelo).fit(M.iloc[tr], y.iloc[tr])
        p[va] = m.predict_proba(M.iloc[va])[:, 1]
    return p


print("\n--- VALIDAÇÃO CRUZADA (OOF, só treino) ---")
linhas = []
for nome, modelo, corrigido, lim in modelos:
    p = oof(modelo, X_treino_c if corrigido else X_treino)
    f1 = f1_score(y, (p >= lim).astype(int))
    auc = roc_auc_score(y, p)
    linhas.append({"modelo": nome, "AUC": round(auc, 4), "F1": round(f1, 4), "limiar": lim})
    print(f"{nome:24s} AUC={auc:.4f}  F1@{lim:.2f}={f1:.4f}")


# filtro de ruído (Brodley & Friedl, 1999): sai do treino quem o RF out-of-bag contradiz com confiança >= 0.8
CORTE_RUIDO = 0.8


def mascara_sem_ruido(M, yy):
    rf = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
                   ('clf', RandomForestClassifier(n_estimators=500, min_samples_leaf=2, oob_score=True,
                                                  random_state=SEED, n_jobs=-1))])
    rf.fit(M, yy)
    p = np.nan_to_num(rf[-1].oob_decision_function_[:, 1], nan=0.5)
    yy = np.asarray(yy)
    ruido = ((yy == 1) & (p < 1 - CORTE_RUIDO)) | ((yy == 0) & (p > CORTE_RUIDO))
    return ~ruido


# filtra só o treino de cada fold; a validação usa todos os pacientes
def oof_sem_ruido(modelo, M, seed):
    p = np.zeros(len(y))
    removidos = []
    for tr, va in gera_folds(seed):
        keep = mascara_sem_ruido(M.iloc[tr], y.iloc[tr])
        removidos.append(int((~keep).sum()))
        idx = tr[keep]
        m = clone(modelo).fit(M.iloc[idx], y.iloc[idx])
        p[va] = m.predict_proba(M.iloc[va])[:, 1]
    return p, float(np.mean(removidos))


# na faixa de dúvida do modelo (0.40 a 0.70) decide o critério clínico da ADA
FAIXA = 0.15


# HbA1c >= 6.5% ou glicemia >= 126 -> 1; HbA1c < 5.7% e glicemia < 100 -> 0; senão, sem decisão
def criterio_ada(M):
    hb = M["HEMOGLOBIN"] if "HEMOGLOBIN" in M else pd.Series(np.nan, index=M.index)
    gl = M["GLUCOSE"] if "GLUCOSE" in M else pd.Series(np.nan, index=M.index)
    alto = (hb >= 6.5) | (gl >= 126)
    normal = (hb.isna() | (hb < 5.7)) & (gl.isna() | (gl < 100)) & (hb.notna() | gl.notna())
    crit = pd.Series(np.nan, index=M.index)
    crit[normal] = 0.0
    crit[alto] = 1.0
    return crit.to_numpy()


def regra_fronteira(p, crit):
    pred = (p >= LIMIAR).astype(int)
    c = np.asarray(crit, dtype=float)
    usar = (np.abs(p - LIMIAR) < FAIXA) & ~np.isnan(c)
    pred[usar] = c[usar].astype(int)
    return pred, int(usar.sum())


crit_tr, crit_te = criterio_ada(X_treino_c), criterio_ada(X_teste_c)
tem = ~np.isnan(crit_tr)

print("\n--- RUÍDO E FRONTEIRA (OOF, 3 repetições) ---")
if tem.any():
    print(f"critério ADA sozinho: decide {tem.mean():.0%} do treino, acerta {np.mean(crit_tr[tem] == y.to_numpy()[tem]):.1%} desses")
R = 3
P = {"corrigido": [oof(voting_c, X_treino_c, SEED + r) for r in range(R)]}
sr = [oof_sem_ruido(voting_c, X_treino_c, SEED + r) for r in range(R)]
P["sem ruído"] = [p for p, _ in sr]
print(f"filtro de ruído: tira em média {np.mean([n for _, n in sr]):.0f} pacientes do treino de cada fold")
f1_var = {}
for base in ("corrigido", "sem ruído"):
    f1_var[base] = float(np.mean([f1_score(y, (p >= LIMIAR).astype(int)) for p in P[base]]))
    regra = [regra_fronteira(p, crit_tr) for p in P[base]]
    f1_var[base + " + regra de fronteira"] = float(np.mean([f1_score(y, pr) for pr, _ in regra]))
    print(f"{base:10s} F1={f1_var[base]:.4f} | com regra de fronteira F1={f1_var[base + ' + regra de fronteira']:.4f}"
          f" ({np.mean([n for _, n in regra]):.0f} pacientes decididos pela regra)")
for nome, v in f1_var.items():
    linhas.append({"modelo": "Voting " + nome + " (3x5-fold)", "AUC": np.nan, "F1": round(v, 4), "limiar": LIMIAR})
pd.DataFrame(linhas).to_csv("resultados_cv.csv", index=False)

# só troca o método base se ganhar pelo menos 1 paciente no OOF
yv = y.to_numpy()
pb = (P["corrigido"][0] >= LIMIAR).astype(int)
tp, fp, fn = int(((pb == 1) & (yv == 1)).sum()), int(((pb == 1) & (yv == 0)).sum()), int(((pb == 0) & (yv == 1)).sum())
GANHO_MIN = 2 * (tp + 1) / (2 * (tp + 1) + fp + fn - 1) - 2 * tp / (2 * tp + fp + fn)
melhor = max(f1_var, key=f1_var.get)
if f1_var[melhor] - f1_var["corrigido"] < GANHO_MIN:
    melhor = "corrigido"
print(f"1 paciente no OOF = {GANHO_MIN:.4f} de F1 -> método escolhido: Voting {melhor}")

# treino final com o método escolhido
print("\n--- TREINO FINAL ---")
A, yA = X_treino_c, y
if melhor.startswith("sem ruído"):
    keep = mascara_sem_ruido(X_treino_c, y)
    idx = np.flatnonzero(keep)
    A, yA = X_treino_c.iloc[idx], y.iloc[idx]
    print(f"filtro de ruído tirou {int((~keep).sum())} pacientes do treino")
probs = clone(voting_c).fit(A, yA).predict_proba(X_teste_c)[:, 1]
if "regra" in melhor:
    preds, n = regra_fronteira(probs, crit_te)
    print(f"regra de fronteira decidiu {n} pacientes do teste")
else:
    preds = (probs >= LIMIAR).astype(int)
pd.DataFrame({"id": ids, "class": preds}).to_csv("submission.csv", index=False)
print(f">>> submission.csv gerado ({preds.mean():.1%} positivos) - é esse que vai pro Kaggle")
