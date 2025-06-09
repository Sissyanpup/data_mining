import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# 1. Baca dataset
df = pd.read_csv('[insert dataset address included dataset file name]')

# 2. Normalisasi label agar huruf depan kapital
df['Esensial'] = df['Esensial'].str.capitalize()

# 3. Label encoding untuk 'Esensial'
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Esensial'])

# 4. Fitur dan target
X = df[['JumlahC', 'JumlahH', 'JumlahO', 'JumlahN', 'JumlahS']]
y = df['Label']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. KNN model
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

# 7. Evaluasi
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n== Evaluasi Model (berbasis massa unsur) ==")
print("Akurasi:", round(accuracy*100, 3), "%")
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
#sns.scatterplot(data=df, x='JumlahC', y='JumlahH', hue='Esensial', palette='Set1', s=100)
#plt.title('Sebaran Jumlah C dan H pada Asam Amino')
#plt.tight_layout()

# Scatter plot semua unsur — misalnya: JumlahC vs JumlahS
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x='JumlahO', y='JumlahH', hue='Esensial', palette='Set2', s=100)
plt.title('Sebaran Jumlah O dan H pada Asam Amino')

# Tambahkan teks evaluasi di pojok kiri bawah
eval_text = f"Akurasi : {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall   : {recall:.2f}\nF1-score : {f1:.2f}"
plt.text(x=df['JumlahC'].min() - 0.5,
         y=df['JumlahS'].min() - 0.2,
         s=eval_text,
         fontsize=10,
         ha='left',
         va='top',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.show()
