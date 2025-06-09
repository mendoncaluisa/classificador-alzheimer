import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_curve,
    roc_auc_score
)

# Carrega o dataset
df = pd.read_csv("alzheimers_disease_data.csv")

# Remove colunas não numéricas
df = df.drop(columns=["PatientID", "DoctorInCharge"])

# Define as labels e o alvo
X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

# Normaliza os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separa treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Cria pipelines para os três classificadores
pipelines = {
    "Naive Bayes": Pipeline([("clf", GaussianNB())]),
    "MLP": Pipeline([("clf", MLPClassifier(random_state=42, max_iter=1000))]),
    "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(random_state=42))])
}

results = {}

for name, pipeline in pipelines.items():
    print(f"\nTreinando e avaliando: {name}")
    # Treino
    pipeline.fit(X_train, y_train)
    # Predições
    y_pred = pipeline.predict(X_test)

    y_proba = None
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    results[name] = {
        "Accuracy": acc,
        "Confusion Matrix": cm,
        "Classification Report": report,
        "AUC": auc
    }

    # Exibir métricas
    print(f"Accuracy: {acc:.4f}")
    print("Matriz de Confusão:")
    print(cm)
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    if auc:
        print(f"AUC: {auc:.4f}")

    # Plot ROC
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

# Curva ROC geral
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.title("Curvas ROC")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("curva_roc_modelos.png", dpi=300)
plt.close()


# 8. Comparação Final
summary = pd.DataFrame({
    name: {
        "Accuracy": r["Accuracy"],
        "Precision": r["Classification Report"]["weighted avg"]["precision"],
        "Recall": r["Classification Report"]["weighted avg"]["recall"],
        "F1-score": r["Classification Report"]["weighted avg"]["f1-score"],
        "AUC": r["AUC"]
    } for name, r in results.items()
}).T

print("\nResumo dos Resultados Comparativos:")
print(summary.round(4))
