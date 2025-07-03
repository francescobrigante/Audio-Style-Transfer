import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

'''To check the variety of traces in one or two datasets'''

# Input directories
dataset1 = r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\PianoDatasetPronto"
dataset2 = r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\ViolinDatasetUnified_44100"  #delete row in case you want to analyze a dataset only

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

# Load features from both datasets 
all_features = []
all_labels = []
all_colors = []

def load_dataset(folder_path, color):
    features = []
    labels = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".mp3"):
            path = os.path.join(folder_path, fname)
            try:
                feat = extract_features(path)
                features.append(feat)
                labels.append(fname)
            except Exception as e:
                print(f"[Error] {fname}: {e}")
    return features, labels, [color] * len(features)

# Dataset 1 (blue points)
feat1, lab1, col1 = load_dataset(dataset1, 'blue')
all_features.extend(feat1)
all_labels.extend(lab1)
all_colors.extend(col1)

# Dataset 2 (red points)
feat2, lab2, col2 = load_dataset(dataset2, 'red')       #delete row in case you want to analyze a dataset only
all_features.extend(feat2)                              #delete row in case you want to analyze a dataset only
all_labels.extend(lab2)                                 #delete row in case you want to analyze a dataset only
all_colors.extend(col2)                                 #delete row in case you want to analyze a dataset only

# Dimensionality reduction (TSNE or PCA) 
all_features = np.array(all_features)
embedding = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(all_features)

# Visualization 
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=all_colors, alpha=0.7)
plt.title("Sonic Diversity Distribution (t-SNE on MFCC Features)")

for i, label in enumerate(all_labels):
    plt.annotate(label, (embedding[i, 0], embedding[i, 1]), fontsize=6, alpha=0.5)

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()