#!/usr/bin/env python3
"""
Detecção de Diabetes - experimento com os casos de fronteira (pesos focais, features KNN e ajuste de limiar)
Igor Oliveira, Leandro Marques e Adiel Pereira

Uso (na pasta com "kaggle-training set" e "kaggle-test"): python3 diabetes_hardmargin.py
Saídas: submission_hardmargin.csv e resultados_hardmargin.csv
"""
import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

SEED = 42
PESOS = {0: 1.0, 1: 1.4}
# pesos focais: de 1x (caso fácil) até 1 + LAMBDA (caso que o modelo erra)
GAMMA, LAMBDA = 2.0, 4.0
K_VIZ = 15
# limiares testados na validação
LIMIARES = np.round(np.arange(0.480, 0.5801, 0.001), 3)
R = 3
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


X_treino_c, X_teste_c, numericas = matrizes(corrige_valores(normaliza_categorias(X)),
                                           corrige_valores(normaliza_categorias(X_test)))


# KNN e MLP: escala só as colunas numéricas (dummies ficam 0/1)
def escalado(clf):
    num = Pipeline([('imputer', SimpleImputer(strategy='median')),
                    ('scaler', QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=SEED))])
    prep = ColumnTransformer([('num', num, numericas),
                              ('faltante', MissingIndicator(error_on_new=False), numericas)],
                             remainder='passthrough')
    return Pipeline([('prep', prep), ('clf', clf)])


# 5-fold estratificado; paciente repetido (PATIENT_CODE) fica sempre no mesmo fold
def gera_folds(seed):
    if "PATIENT_CODE" in train and train["PATIENT_CODE"].dropna().duplicated().any():
        grupos = train["PATIENT_CODE"].fillna(pd.Series([f"sem_{i}" for i in range(len(train))], index=train.index))
        return list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X_treino_c, y, grupos))
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_treino_c, y))


# Voting RF + ET + HGB (pesos 1.2, 1.2, 1.0) aceitando sample_weight
def treina_voting(Xtr, ytr, w=None):
    imp999 = SimpleImputer(strategy='constant', fill_value=-999).fit(Xtr)
    impmed = SimpleImputer(strategy='median', add_indicator=True).fit(Xtr)
    A999, Amed = imp999.transform(Xtr), impmed.transform(Xtr)
    rf = RandomForestClassifier(n_estimators=600, max_depth=12, class_weight=PESOS, random_state=SEED,
                                n_jobs=-1).fit(A999, ytr, sample_weight=w)
    et = ExtraTreesClassifier(n_estimators=600, max_depth=12, class_weight=PESOS, random_state=SEED,
                              n_jobs=-1).fit(Amed, ytr, sample_weight=w)
    hgb = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=500, max_depth=5, min_samples_leaf=10,
                                         l2_regularization=0.1, class_weight='balanced',
                                         random_state=SEED).fit(A999, ytr, sample_weight=w)

    def proba(Xv):
        V999, Vmed = imp999.transform(Xv), impmed.transform(Xv)
        return (1.2 * rf.predict_proba(V999)[:, 1] + 1.2 * et.predict_proba(Vmed)[:, 1]
                + 1.0 * hgb.predict_proba(V999)[:, 1]) / 3.4
    return proba


# peso = 1 + LAMBDA * (1 - p)^GAMMA, com p = probabilidade out-of-bag da classe verdadeira
def pesos_focais(Xtr, ytr):
    A = SimpleImputer(strategy='constant', fill_value=-999).fit_transform(Xtr)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, oob_score=True, random_state=SEED,
                                n_jobs=-1).fit(A, ytr)
    p = np.nan_to_num(rf.oob_decision_function_[:, 1], nan=0.5)
    yy = np.asarray(ytr)
    p_t = np.where(yy == 1, p, 1 - p)
    return 1.0 + LAMBDA * (1.0 - p_t) ** GAMMA


# fração de diabéticos entre os K vizinhos e distância ao diabético e ao saudável mais próximos (sem o próprio paciente)
def features_knn(Xref, yref, Xq=None):
    prep = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('qt', QuantileTransformer(output_distribution='normal', n_quantiles=min(200, len(Xref)),
                                                random_state=SEED))])
    Rf = prep.fit_transform(Xref[numericas])
    proprio = Xq is None
    Q = Rf if proprio else prep.transform(Xq[numericas])
    yr = np.asarray(yref)
    if proprio:
        _, idx = NearestNeighbors(n_neighbors=K_VIZ + 1).fit(Rf).kneighbors(Q)
        idx = np.array([linha[linha != i][:K_VIZ] for i, linha in enumerate(idx)])
    else:
        _, idx = NearestNeighbors(n_neighbors=K_VIZ).fit(Rf).kneighbors(Q)
    out = {"KNN_FRAC_DIAB": yr[idx].mean(axis=1)}
    for classe, nome in ((1, "KNN_DIST_DIAB"), (0, "KNN_DIST_SAUD")):
        d, _ = NearestNeighbors(n_neighbors=2).fit(Rf[yr == classe]).kneighbors(Q)
        out[nome] = np.where(yr == classe, d[:, 1], d[:, 0]) if proprio else d[:, 0]
    out["KNN_RAZAO"] = out["KNN_DIST_SAUD"] / (out["KNN_DIST_DIAB"] + out["KNN_DIST_SAUD"] + 1e-9)
    return pd.DataFrame(out, index=(Xref if proprio else Xq).index)


# pesos e features KNN calculados só com o treino recebido
def monta(Xtr, ytr, Xv, hard):
    if not hard:
        return Xtr, Xv, None
    w = pesos_focais(Xtr, ytr)
    A = pd.concat([Xtr, features_knn(Xtr, ytr)], axis=1)
    B = pd.concat([Xv, features_knn(Xtr, ytr, Xv)], axis=1)
    return A, B, w


def oof_variante(seed, hard):
    p = np.zeros(len(y))
    for tr, va in gera_folds(seed):
        A, B, w = monta(X_treino_c.iloc[tr], y.iloc[tr], X_treino_c.iloc[va], hard)
        p[va] = treina_voting(A, y.iloc[tr], w)(B)
    return p


# F1 médio das repetições para cada limiar
def curva_f1(Ps):
    yb = y.to_numpy().astype(bool)
    curvas = []
    for p in Ps:
        pred = p[None, :] >= LIMIARES[:, None]
        tp, fp, fn = (pred & yb).sum(1), (pred & ~yb).sum(1), (~pred & yb).sum(1)
        curvas.append(2 * tp / np.maximum(2 * tp + fp + fn, 1))
    return np.mean(curvas, axis=0)


# modelos de partida exigidos pelo trabalho
linhas = []
print("\n--- MODELOS EXIGIDOS (OOF 5-fold, só treino) ---")
exigidos = [("KNN", escalado(KNeighborsClassifier(n_neighbors=15, weights='distance'))),
            ("Arvore de Decisao", Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
                                            ('clf', DecisionTreeClassifier(max_depth=6, min_samples_leaf=10,
                                                                           class_weight=PESOS, random_state=SEED))])),
            ("MLP", escalado(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=1000,
                                           early_stopping=True, random_state=SEED)))]
for nome, modelo in exigidos:
    p = np.zeros(len(y))
    for tr, va in gera_folds(SEED):
        p[va] = clone(modelo).fit(X_treino_c.iloc[tr], y.iloc[tr]).predict_proba(X_treino_c.iloc[va])[:, 1]
    f1, auc = f1_score(y, (p >= 0.5).astype(int)), roc_auc_score(y, p)
    linhas.append({"modelo": nome, "AUC": round(auc, 4), "F1": round(f1, 4), "limiar": 0.5})
    print(f"{nome:22s} AUC={auc:.4f}  F1@0.50={f1:.4f}")

print(f"\n--- VOTING x HARD-MARGIN (OOF {R}x 5-fold, limiar varrido de 0.480 a 0.580) ---")
# Voting x Voting com pesos focais e KNN, cada um no melhor limiar do OOF
res = {}
for nome, hard in (("Voting", False), ("Voting hard-margin", True)):
    Ps = [oof_variante(SEED + r, hard) for r in range(R)]
    c = curva_f1(Ps)
    i = int(np.argmax(c))
    f055 = float(c[np.argmin(np.abs(LIMIARES - 0.55))])
    auc = float(np.mean([roc_auc_score(y, p) for p in Ps]))
    res[nome] = (float(LIMIARES[i]), float(c[i]))
    linhas.append({"modelo": nome, "AUC": round(auc, 4), "F1": round(float(c[i]), 4), "limiar": float(LIMIARES[i])})
    print(f"{nome:22s} AUC={auc:.4f}  F1@0.550={f055:.4f}  melhor F1={c[i]:.4f} @ {LIMIARES[i]:.3f}")
pd.DataFrame(linhas).to_csv("resultados_hardmargin.csv", index=False)

escolhido = max(res, key=lambda k: res[k][1])
lim = res[escolhido][0]
print(f"método escolhido no OOF: {escolhido} @ {lim:.3f}")

print("\n--- TREINO FINAL ---")
# treino final com o método escolhido
A, B, w = monta(X_treino_c, y, X_teste_c, escolhido.endswith("hard-margin"))
if w is not None:
    print(f"pesos focais no treino: {np.mean(w > 2):.0%} dos pacientes com peso > 2 (máx {w.max():.1f})")
probs = treina_voting(A, y, w)(B)
preds = (probs >= lim).astype(int)
pd.DataFrame({"id": ids, "class": preds}).to_csv("submission_hardmargin.csv", index=False)
print(f">>> submission_hardmargin.csv gerado ({preds.mean():.1%} positivos)")
