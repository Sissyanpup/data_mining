import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import mode
from scipy.stats import mode
import numpy as np

# 1. Baca dataset
df = pd.read_csv('[insert dataset address included dataset file name]')

# 2. Normalisasi label
df['Esensial'] = df['Esensial'].str.capitalize()

# 3. Label encoding untuk evaluasi (bukan untuk training model K-Means)
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Esensial'])  # Tidak=0, Ya=1

# 4. Fitur yang digunakan
X = df[['JumlahC', 'JumlahH', 'JumlahO', 'JumlahN', 'JumlahS']]

# 5. K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# 6. Mapping cluster ke label manual (agar cluster 0/1 sesuai label)
# Caranya: lihat mana cluster yang dominan untuk label 1 dan 0
def cluster_to_label_mapping(y_true, y_clusters):
    labels = np.zeros_like(y_clusters)
    for i in np.unique(y_clusters):
        mask = y_clusters == i
        majority_label = mode(y_true[mask], keepdims=True)[0][0]
        labels[mask] = majority_label
    return labels

mapped_cluster = cluster_to_label_mapping(df['Label'].values, df['Cluster'].values)

# Evaluasi hasil clustering terhadap label asli
akurasi = accuracy_score(df['Label'], mapped_cluster)

print("\n=== Evaluasi Clustering K-Means terhadap Label Esensial ===")
print("Akurasi (Purity):", round(akurasi * 100, 2), "%")
print("Confusion Matrix:")
print(confusion_matrix(df['Label'], mapped_cluster))

# 7. Visualisasi

sns.set(style="whitegrid")

# Visualisasi cluster dalam ruang 2D (gunakan C dan N sebagai contoh)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='JumlahC', y='JumlahN', hue='Cluster', palette='Set1', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 3], 
            s=200, c='black', marker='X', label='Centroids')
plt.title("Cluster Asam Amino (K-Means)")
plt.xlabel("Jumlah C")
plt.ylabel("Jumlah N")
plt.legend()
plt.tight_layout()

# Distribusi cluster vs label
plt.figure(figsize=(6, 4))
sns.countplot(x='Cluster', hue='Esensial', data=df, palette='Set2')
plt.title("Distribusi Cluster terhadap Label Esensial")
plt.tight_layout()

# Rata-rata fitur per cluster
mean_fitur = df.groupby('Cluster')[['JumlahC', 'JumlahH', 'JumlahO', 'JumlahN', 'JumlahS']].mean().reset_index()
mean_fitur_melted = mean_fitur.melt(id_vars='Cluster', var_name='Unsur', value_name='Rata-rata Massa')

plt.figure(figsize=(8, 5))
sns.barplot(data=mean_fitur_melted, x='Unsur', y='Rata-rata Massa', hue='Cluster', palette='Set1')
plt.title("Rata-rata Unsur per Cluster")
plt.tight_layout()


# 7. Evaluasi clustering
accuracy = accuracy_score(df['Label'], mapped_cluster)
precision = precision_score(df['Label'], mapped_cluster)
recall = recall_score(df['Label'], mapped_cluster)
f1 = f1_score(df['Label'], mapped_cluster)
conf_matrix = confusion_matrix(df['Label'], mapped_cluster)

# 8. Tampilkan metrik evaluasi
print("\n=== Evaluasi K-Means Clustering ===")
print(f"Akurasi  : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1-Score : {f1:.2f}")

# 9. Visualisasi confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix K-Means")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
