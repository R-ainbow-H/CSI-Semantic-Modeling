import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score
)

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist


from util import load_data_n_model



# ==================================================
# 1. Configuration
# ==================================================

dataset_name = "NTU-Fi_HAR"
model_name = "ResNet18"

checkpoint_path = "./weights/best_resnet18.pth"

save_dir = "embedding_analysis"

os.makedirs(save_dir, exist_ok=True)


class_names = [
    "box",
    "circle",
    "clean",
    "fall",
    "run",
    "walk"
]



# ==================================================
# 2. Load Dataset and Model
# ==================================================

print("="*60)
print("Loading dataset and model")
print("="*60)


train_loader, test_loader, model, train_epoch = load_data_n_model(
    dataset_name,
    model_name
)


model.eval()


# load trained weights

print("Loading checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)

model.load_state_dict(checkpoint)

print("Model loaded successfully")



# ==================================================
# 3. Build Feature Extractor
# Remove FC classifier
# ==================================================

feature_extractor = nn.Sequential(

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


feature_extractor.eval()


print("Feature extractor ready")



# ==================================================
# 4. Extract 512-d Features
# ==================================================

features = []
labels = []


print("\nExtracting features...")


with torch.no_grad():

    for idx,(x,y) in enumerate(train_loader):

        embedding = feature_extractor(x)


        embedding = embedding.reshape(
            embedding.shape[0],
            -1
        )


        features.append(
            embedding.cpu().numpy()
        )

        labels.append(
            y.numpy()
        )


        print(
            f"Batch {idx+1}/{len(train_loader)} finished"
        )



features = np.concatenate(
    features,
    axis=0
)


labels = np.concatenate(
    labels,
    axis=0
)



print("\nFeature extraction finished")

print(
    "Feature shape:",
    features.shape
)

print(
    "Label shape:",
    labels.shape
)



np.save(
    f"{save_dir}/features.npy",
    features
)

np.save(
    f"{save_dir}/labels.npy",
    labels
)



# ==================================================
# 5. Embedding Statistics
# ==================================================

print("\n")
print("="*60)
print("Embedding Statistics")
print("="*60)


mean_value = np.mean(features)
std_value = np.std(features)


print("Mean:", mean_value)
print("Std :", std_value)
print("Max :", np.max(features))
print("Min :", np.min(features))



# ==================================================
# 6. Silhouette Score
# ==================================================

print("\n")
print("="*60)
print("Cluster Evaluation")
print("="*60)



sil_score = silhouette_score(
    features,
    labels
)


print(
    "Silhouette Score:",
    sil_score
)



# ==================================================
# 7. KMeans clustering
# ==================================================

print("\nRunning KMeans...")


kmeans = KMeans(
    n_clusters=len(class_names),
    random_state=42,
    n_init=10
)


cluster_labels = kmeans.fit_predict(
    features
)


ari = adjusted_rand_score(
    labels,
    cluster_labels
)


print(
    "ARI:",
    ari
)



# ==================================================
# 8. Calculate Centroids
# ==================================================

print("\nCalculating centroids...")


centroids=[]


for i,name in enumerate(class_names):

    cls_feature = features[
        labels==i
    ]

    centroid = np.mean(
        cls_feature,
        axis=0
    )

    centroids.append(
        centroid
    )


    print(
        name,
        cls_feature.shape
    )


centroids=np.array(
    centroids
)


print(
    "Centroid shape:",
    centroids.shape
)



# ==================================================
# 9. Cosine Similarity Matrix
# ==================================================

print("\nCosine similarity")


norm = np.linalg.norm(
    centroids,
    axis=1,
    keepdims=True
)


normalized_centroid = (
    centroids / norm
)


similarity_matrix = (
    normalized_centroid
    @
    normalized_centroid.T
)



print(similarity_matrix)


np.savetxt(
    f"{save_dir}/similarity.csv",
    similarity_matrix,
    delimiter=","
)



# ==================================================
# 10. Similarity Heatmap
# ==================================================

plt.figure(figsize=(7,6))


plt.imshow(
    similarity_matrix
)


plt.colorbar()


plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)


plt.yticks(
    range(len(class_names)),
    class_names
)


plt.title(
    "Cosine Similarity Matrix"
)


plt.tight_layout()


plt.savefig(
    f"{save_dir}/similarity_heatmap.png",
    dpi=300
)


plt.close()



# ==================================================
# 11. Intra-class Distance
# ==================================================

print("\nIntra-class distance")


intra_distance=[]


for i,name in enumerate(class_names):

    cls_feature = features[
        labels==i
    ]

    centroid = centroids[i]


    distance = np.mean(
        np.linalg.norm(
            cls_feature-centroid,
            axis=1
        )
    )


    intra_distance.append(
        distance
    )


    print(
        name,
        distance
    )



# ==================================================
# 12. Inter-class Distance
# ==================================================

inter_distance = cdist(
    centroids,
    centroids,
    metric="euclidean"
)


print(
    "\nInter-class distance:"
)

print(
    inter_distance
)


np.savetxt(
    f"{save_dir}/distance_matrix.csv",
    inter_distance,
    delimiter=","
)



# ==================================================
# 13. PCA
# ==================================================

print("\nRunning PCA...")


pca=PCA(
    n_components=2
)


feature_pca=pca.fit_transform(
    features
)


plt.figure(figsize=(8,6))


for i,name in enumerate(class_names):

    idx = labels==i

    plt.scatter(
        feature_pca[idx,0],
        feature_pca[idx,1],
        s=15,
        label=name
    )


plt.legend()

plt.title(
    "PCA of CSI Embedding"
)


plt.savefig(
    f"{save_dir}/pca.png",
    dpi=300
)


plt.close()



# ==================================================
# 14. t-SNE
# ==================================================

print("\nRunning t-SNE...")


tsne=TSNE(
    n_components=2,
    random_state=42,
    init="pca"
)


feature_tsne=tsne.fit_transform(
    features
)



plt.figure(figsize=(8,6))


for i,name in enumerate(class_names):

    idx=labels==i

    plt.scatter(
        feature_tsne[idx,0],
        feature_tsne[idx,1],
        s=15,
        label=name
    )


plt.legend()


plt.title(
    "t-SNE of CSI Embedding"
)


plt.savefig(
    f"{save_dir}/tsne.png",
    dpi=300
)


plt.close()



# ==================================================
# 15. Save Report
# ==================================================

report_path = f"{save_dir}/report.txt"


with open(
    report_path,
    "w"
) as f:

    f.write(
        "CSI Embedding Analysis Report\n\n"
    )


    f.write(
        f"Feature shape: {features.shape}\n"
    )


    f.write(
        f"Silhouette Score: {sil_score}\n"
    )


    f.write(
        f"ARI: {ari}\n\n"
    )


    f.write(
        "Intra-class distance:\n"
    )


    for name,d in zip(
        class_names,
        intra_distance
    ):

        f.write(
            f"{name}: {d}\n"
        )



print("\n")
print("="*60)
print("Analysis Finished")
print("="*60)

print(
    "Results saved in:",
    save_dir
)