import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc,
                             RocCurveDisplay)
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.multiclass import OneVsRestClassifier

data = pd.read_csv("alzheimers_disease_data.csv")
# Vou assumir que a última coluna é o target e as outras são features
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Codificar labels se for categórico
if isinstance(y[0], str):
    le = LabelEncoder()
    y = le.fit_transform(y)
    class_names = le.classes_
else:
    class_names = np.unique(y)

# Binarizar as labels para plotar a curva ROC
y_bin = label_binarize(y, classes=np.unique(y))
n_classes = y_bin.shape[1]

# Dividir em treino e teste
X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
    X, y, y_bin, test_size=0.3, random_state=42
)


# Função para plotar curva ROC multiclasse
def plot_multiclass_roc(y_test_bin, y_score, n_classes, class_names, title):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'][:n_classes])
    plt.figure(figsize=(8, 6))

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

# 1. Pipeline para MLPClassifier
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))
])

mlp_pipeline.fit(X_train, y_train)
y_pred_mlp = mlp_pipeline.predict(X_test)
y_score_mlp = mlp_pipeline.predict_proba(X_test)

print("\n=== MLPClassifier ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred_mlp):.4f}")
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred_mlp))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_mlp, target_names=class_names))

plot_multiclass_roc(y_test_bin, y_score_mlp, n_classes, class_names, "ROC Curve - MLPClassifier")

# 2. Pipeline para GaussianNB
nb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('nb', GaussianNB())
])

nb_pipeline.fit(X_train, y_train)
y_pred_nb = nb_pipeline.predict(X_test)
y_score_nb = nb_pipeline.predict_proba(X_test)

print("\n=== GaussianNB ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred_nb):.4f}")
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred_nb))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_nb, target_names=class_names))

plot_multiclass_roc(y_test_bin, y_score_nb, n_classes, class_names, "ROC Curve - GaussianNB")

# 3. Pipeline para DecisionTreeClassifier
dt_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('dt', DecisionTreeClassifier(random_state=42))
])

dt_pipeline.fit(X_train, y_train)
y_pred_dt = dt_pipeline.predict(X_test)
y_score_dt = dt_pipeline.predict_proba(X_test)

print("\n=== DecisionTreeClassifier ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred_dt):.4f}")
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred_dt))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_dt, target_names=class_names))

plot_multiclass_roc(y_test_bin, y_score_dt, n_classes, class_names, "ROC Curve - DecisionTreeClassifier")


