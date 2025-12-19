import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    r2_score,classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

dk = pd.read_csv(r"C:\Users\hp\Downloads\AI ML\project\creditcard.csv")
df = dk.copy()

cols = ["V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14",
        "V15","V16","V17","V18","V19","V20","V21","V22","V23","V24",
        "V25","V26","V27","V28","Amount"]

df[cols] = df[cols].fillna(df[cols].mean())
df = df.dropna(subset = ["Class"])

fraud_ratio = df['Class'].mean()
print(f"Fraud ratio: {fraud_ratio:.6f} ({fraud_ratio*100:.4f}%)")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(df[df["Class"] == 0]["Amount"], bins=50)
plt.title("Amount(not fraud)")
plt.subplot(1,2,2)
plt.hist(df[df["Class"] == 1]["Amount"], bins=50)
plt.title("Amount(fraud)")
plt.show()

Column = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14",
        "V15","V16","V17","V18","V19","V20","V21","V22","V23","V24",
        "V25","V26","V27","V28","Amount"]


X = df[Column]
Y = df["Class"]

X_trainval,X_test,Y_trainval,Y_test = train_test_split(X,Y,random_state = 42,
                                                 stratify = Y,test_size =0.2)

X_train,X_val,Y_train,Y_val = train_test_split(X_trainval,Y_trainval,test_size = 0.25,
                                               stratify = Y_trainval,random_state = 42)

print(Counter(Y))

pipe = Pipeline([
    ("scaler",StandardScaler()),
    ("lr_model",LogisticRegression(max_iter = 100)),
])

pipe.fit(X_train,Y_train)

lr_pred = pipe.predict(X_test)
print(lr_pred)
print(classification_report(Y_test,lr_pred))
print(confusion_matrix(Y_test,lr_pred))


print("R2(Logistic Regression) :",r2_score(Y_test,lr_pred))

pipe_rf = Pipeline([
    ("scaler",StandardScaler()),
    ("lr_model",RandomForestClassifier(n_estimators = 200 , n_jobs = -1 ,
                                       random_state = 42)),
])

pipe_rf.fit(X_train,Y_train)

rf_pred = pipe_rf.predict(X_test)
print(rf_pred)
print(classification_report(Y_test,rf_pred))
print(confusion_matrix(Y_test,rf_pred))
print("R2(Random Forest Classifier) :",r2_score(Y_test,rf_pred))

probs = pipe.predict_proba(X_test)[:,1]
print(probs)

print("ROC AUC:", roc_auc_score(Y_test, probs))

print("Average Precision (PR AUC):", average_precision_score(Y_test, probs))

smote = SMOTE(random_state=42)
pipe_smote = ImbPipeline(steps=[
    ('pre', StandardScaler()),
    ('sm', smote),
    ('clf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
])

pipe_smote.fit(X_train, Y_train)
probs_sm = pipe_smote.predict_proba(X_test)[:,1]
preds_sm = pipe_smote.predict(X_test)

print("\nRandomForest (SMOTE) — Validation metrics:")
print(classification_report(Y_test, preds_sm, digits=4))
print("ROC AUC:", roc_auc_score(Y_test, probs_sm))
print("Average Precision (PR AUC):", average_precision_score(Y_test, probs_sm))

print(r2_score(Y_test,preds_sm))

precision,recall,thresholds = precision_recall_curve(Y_test,probs_sm)
plt.plot(precision,recall)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("PR CURVE")
plt.show()

f1_scores = 2*precision*recall/(precision +recall+1e-12)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
print("Best F1 threshold (val):", best_threshold, "F1:", f1_scores[best_idx])

Y_full_train = pd.concat([Y_train,Y_val])
X_full_train = pd.concat([X_train,X_val])
smote  = SMOTE(random_state = 42)
imb_pipe = ImbPipeline(steps =[
("scale",StandardScaler()),
("sm",smote),
("model",RandomForestClassifier(n_estimators = 50,n_jobs = -1,random_state = 42))                                                      
])

imb_pipe.fit(X_full_train,Y_full_train)
imb_probs = imb_pipe.predict_proba(X_test)[:,1]
imb_pred = imb_pipe.predict(X_test)
imb_pred_thres = (imb_probs > best_threshold).astype(int)

print("\nRandomForest (SMOTE) — Validation metrics:")
print(classification_report(Y_test,imb_pred,digits = 2))
print("confusion matrix:",confusion_matrix(Y_test,imb_pred))
print("roc_auc_score:",roc_auc_score(Y_test,imb_probs))
print("average_precision_score:",average_precision_score(Y_test,imb_probs))
print("precision:",precision_score(imb_pred,imb_pred_thres))
print("recall:",recall_score(imb_pred,imb_pred_thres))
print("f1_score:",f1_score(imb_pred,imb_pred_thres))

cm = confusion_matrix(Y_test,imb_pred)
plt.figure(figsize =(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("Confusion Matrix")
plt.show()

fpr,tpr,_ = roc_curve(Y_test,imb_probs)
precision,recall,thresholds = precision_recall_curve(Y_test,imb_probs)

plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
plt.plot(fpr,tpr);plt.xlabel("FPR");plt.ylabel("TPR");plt.title("ROC CURVE")
plt.subplot(1,2,2);plt.plot(precision,recall);plt.xlabel("Precision");plt.ylabel("Recall");plt.title("PR CURVE")
plt.show()

# CROSS VALIDATION
param_dist = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 8, 16, 32],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

pipe_for_search = ImbPipeline(steps=[
    ('scale', StandardScaler()),
    ('sm', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
])

rs = RandomizedSearchCV(pipe_for_search, param_distributions=param_dist, n_iter=12,
                        scoring='average_precision', cv=StratifiedKFold(n_splits=3),
                        verbose=2, n_jobs=-1, random_state=42)
rs.fit(X_full_train, Y_full_train)
print("Best params:", rs.best_params_)
