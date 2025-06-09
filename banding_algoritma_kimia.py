import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# 1. Load dataset
df = pd.read_csv("[insert dataset address included dataset file name]")

# 2. Preprocessing
df['Esensial'] = df['Esensial'].str.capitalize()
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Esensial'])

X = df[['JumlahC', 'JumlahH', 'JumlahO', 'JumlahN', 'JumlahS']]
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "KMeans": make_pipeline(StandardScaler(), KMeans(n_clusters=2, random_state=42))
}

results = []
fpr_tpr = {}

for name, model in models.items():
    if name == "KMeans":
        model.fit(X_train)
        y_pred = model.predict(X_test)
        if roc_auc_score(y_test, y_pred) < 0.5:
            y_pred = 1 - y_pred
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    fpr_tpr[name] = (fpr, tpr)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

# 4. Convert results to DataFrame
results_df = pd.DataFrame(results)

# 5. Plot ROC Curve
plt.figure(figsize=(10, 6))
for name, (fpr, tpr) in fpr_tpr.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={results_df[results_df['Model'] == name]['ROC AUC'].values[0]:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Perbandingan Model')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Plot Heatmap Evaluasi
plt.figure(figsize=(10, 4))
sns.heatmap(
    results_df.drop(columns="Model").set_index(results_df['Model']).T,
    annot=True, cmap="YlGnBu", fmt=".2f", cbar=True
)
plt.title("Perbandingan Evaluasi Model")
plt.tight_layout()
plt.show()
