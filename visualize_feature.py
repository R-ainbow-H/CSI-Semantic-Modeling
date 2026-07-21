import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from util import load_data_n_model


# ===============================
# 1. Load dataset and model
# ===============================
root = "./Data/"

train_loader, test_loader, model, train_epoch = load_data_n_model(
    "NTU-Fi_HAR",
    "ResNet18",
    root
)

# 读取训练好的模型
model.load_state_dict(
    torch.load("weights/final_resnet18.pth", map_location="cpu")
)

model.eval()

print("=" * 60)
print("Feature Visualization")
print("=" * 60)


# ===============================
# 2. Remove classifier (FC)
# ===============================
feature_extractor = torch.nn.Sequential(
    model.reshape,
    model.conv1,
    model.batch_norm1,
    model.relu,
    model.max_pool,
    model.layer1,
    model.layer2,
    model.layer3,
    model.layer4,
    model.avgpool
)


# ===============================
# 3. Extract Features
# ===============================
features = []
labels = []

with torch.no_grad():

    for batch_idx, (x, y) in enumerate(train_loader):

        embedding = feature_extractor(x)

        embedding = embedding.view(embedding.size(0), -1)

        features.append(embedding.cpu().numpy())
        labels.append(y.cpu().numpy())

        print(f"Processed Batch {batch_idx+1}/{len(train_loader)}")

features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

print("\nFinished Feature Extraction")
print("Feature Shape :", features.shape)
print("Label Shape   :", labels.shape)


# ===============================
# 4. Save Feature
# ===============================
os.makedirs("features", exist_ok=True)

np.save("features/features.npy", features)
np.save("features/labels.npy", labels)

print("Saved feature.npy")
print("Saved labels.npy")


# ===============================
# 5. PCA
# ===============================
print("\nRunning PCA...")

pca = PCA(n_components=2)

feature_pca = pca.fit_transform(features)

print("PCA Done")


# ===============================
# 6. Plot PCA
# ===============================
class_names = [
    "box",
    "circle",
    "clean",
    "fall",
    "run",
    "walk"
]

plt.figure(figsize=(8,6))

for i in range(6):
    idx = labels == i
    plt.scatter(
        feature_pca[idx,0],
        feature_pca[idx,1],
        s=15,
        label=class_names[i]
    )

plt.legend()
plt.title("PCA of NTU-Fi Features")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()

plt.savefig(
    "features/pca.png",
    dpi=300
)

print("Saved pca.png")


# ===============================
# 7. t-SNE
# ===============================
print("\nRunning t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca"
)

feature_tsne = tsne.fit_transform(features)

print("t-SNE Done")


# ===============================
# 8. Plot t-SNE
# ===============================
plt.figure(figsize=(8,6))

for i in range(6):
    idx = labels == i
    plt.scatter(
        feature_tsne[idx,0],
        feature_tsne[idx,1],
        s=15,
        label=class_names[i]
    )

plt.legend()

plt.title("t-SNE of NTU-Fi Features")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.tight_layout()

plt.savefig(
    "features/tsne.png",
    dpi=300
)

print("Saved tsne.png")

plt.show()

print("=" * 60)
print("Finished!")
print("=" * 60)