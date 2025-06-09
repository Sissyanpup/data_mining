import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# 1. Baca dataset
df = pd.read_csv('[insert dataset address included dataset file name]')

# 2. Normalisasi label agar huruf depan kapital
df['Esensial'] = df['Esensial'].str.capitalize()  # 'ya' -> 'Ya', 'tidak' -> 'Tidak'

# 3. Label encoding untuk 'Esensial'
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Esensial'])  # ya=1, tidak=0

# 4. Fitur dan target
X = df[['JumlahC', 'JumlahH', 'JumlahO', 'JumlahN', 'JumlahS']]
y = df['Label']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Decision Tree model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 7. Evaluasi
y_pred = clf.predict(X_test)
print("\n== Evaluasi Model (berbasis massa unsur) ==")
print("Akurasi:", round(accuracy_score(y_test, y_pred)*100,3),"%")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 8. Visualisasi
sns.set(style="whitegrid")

# Distribusi label
plt.figure(figsize=(6, 4))
sns.countplot(x='Esensial', data=df, palette='Set2')
plt.title('Distribusi Asam Amino Esensial vs Tidak')
plt.tight_layout()

# Rata-rata massa unsur
mean_massa = df.groupby('Esensial')[['JumlahC', 'JumlahH', 'JumlahO', 'JumlahN', 'JumlahS']].mean().reset_index()
mean_massa_melted = mean_massa.melt(id_vars='Esensial', var_name='Unsur', value_name='Rata-rata Massa')
plt.figure(figsize=(8, 5))
sns.barplot(data=mean_massa_melted, x='Unsur', y='Rata-rata Massa', hue='Esensial', palette='Set1')
plt.title('Rata-rata Massa Unsur Asam Amino')
plt.tight_layout()

# Scatter plot C vs N
#plt.figure(figsize=(6, 5))
#sns.scatterplot(data=df, x='JumlahC', y='JumlahN', hue='Esensial', palette='Set1', s=100)
#plt.title('Sebaran Jumlah C dan N pada Asam Amino')
#plt.tight_layout()
#plt.show()

from sklearn.tree import plot_tree
from sklearn.tree import export_text

# Menampilkan struktur Decision Tree dalam bentuk teks
tree_rules = export_text(clf, feature_names=list(X.columns))
print("\n=== Struktur Decision Tree ===")
print(tree_rules)

from sklearn.metrics import precision_score, recall_score, f1_score

# Hitung metrik evaluasi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Plot pohon dan tambahkan teks evaluasi
plt.figure(figsize=(12, 6))
plot_tree(clf, feature_names=X.columns, class_names=label_encoder.classes_, filled=True, rounded=True)
plt.title("Visualisasi Decision Tree")

# Tambahkan nilai evaluasi di plot
eval_text = f"Akurasi: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1-score: {f1:.2f}"
plt.text(0.0, 0, eval_text, fontsize=12, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

plt.tight_layout()
plt.show()

from sklearn.tree import export_dict

tree_dict = clf.tree_
print("\nJumlah node:", tree_dict.node_count)

