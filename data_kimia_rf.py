import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# 1. Baca dataset
df = pd.read_csv('[insert dataset address included dataset file name]')

# 2. Normalisasi label agar huruf depan kapital
df['Esensial'] = df['Esensial'].str.capitalize()

# 3. Label encoding untuk 'Esensial'
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Esensial'])  # ya=1, tidak=0

# 4. Fitur dan target
X = df[['JumlahC', 'JumlahH', 'JumlahO', 'JumlahN', 'JumlahS']]
y = df['Label']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Random Forest model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# 7. Evaluasi
y_pred = rf_clf.predict(X_test)

print("\n== Evaluasi Model Random Forest ==")
print("Akurasi:", round(accuracy_score(y_test, y_pred)*100, 3), "%")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Hitung metrik evaluasi tambahan
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 8. Visualisasi

sns.set(style="whitegrid")

# Distribusi label
plt.figure(figsize=(6, 4))
sns.countplot(x='Esensial', data=df, palette='Set2')
plt.title('Distribusi Asam Amino Esensial vs Tidak')
plt.tight_layout()

# Rata-rata unsur
mean_massa = df.groupby('Esensial')[['JumlahC', 'JumlahH', 'JumlahO', 'JumlahN', 'JumlahS']].mean().reset_index()
mean_massa_melted = mean_massa.melt(id_vars='Esensial', var_name='Unsur', value_name='Rata-rata Massa')
plt.figure(figsize=(8, 5))
sns.barplot(data=mean_massa_melted, x='Unsur', y='Rata-rata Massa', hue='Esensial', palette='Set1')
plt.title('Rata-rata Massa Unsur Asam Amino')
plt.tight_layout()

# Feature Importance
importances = rf_clf.feature_importances_
feat_names = X.columns
feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

plt.figure(figsize=(7, 4))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()

# Visualisasi salah satu pohon dari Random Forest
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 6))
plot_tree(rf_clf.estimators_[0], feature_names=feat_names,
    class_names=label_encoder.classes_, filled=True, rounded=True)
plt.title("Visualisasi Salah Satu Pohon dalam Random Forest")

# Tambahkan evaluasi di plot
eval_text = f"Akurasi: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1-score: {f1:.2f}"
plt.text(0.8, 1, eval_text, fontsize=12, ha='left', va='top',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

plt.tight_layout()
plt.show()
