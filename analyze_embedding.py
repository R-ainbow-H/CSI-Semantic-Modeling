import os
import numpy as np
import torch

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

from util import load_data_n_model


# =====================================
# 1. Load dataset and trained model
# =====================================

root = "./Data/"

print("Loading dataset...")

train_loader, test_loader, model, train_epoch = load_data_n_model(
    "NTU-Fi_HAR",
    "ResNet18",
    root
)


print("Loading weights...")

model.load_state_dict(
    torch.load(
        "weights/best_resnet18.pth",
        map_location="cpu"
    )
)


model.eval()


print("Model loaded successfully")


# =====================================
# 2. Build feature extractor
# =====================================


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


feature_extractor.eval()


print("Feature extractor ready")


# =====================================
# 3. Extract 512-d embeddings
# =====================================


features = []
labels = []


print("\nExtracting features...")


with torch.no_grad():

    for batch_idx, (x, y) in enumerate(train_loader):

        embedding = feature_extractor(x)


        # (B,512,1,1)
        embedding = embedding.reshape(
            embedding.shape[0],
            -1
        )


        features.append(
            embedding.cpu().numpy()
        )

        labels.append(
            y.cpu().numpy()
        )


        print(
            f"Batch {batch_idx+1}/{len(train_loader)} finished"
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



# =====================================
# 4. Save embeddings
# =====================================


os.makedirs(
    "features",
    exist_ok=True
)


np.save(
    "features/embedding.npy",
    features
)


np.save(
    "features/embedding_labels.npy",
    labels
)


print("Embedding saved")



# =====================================
# 5. Calculate class centroid
# =====================================


class_names = [

    "box",
    "circle",
    "clean",
    "fall",
    "run",
    "walk"

]


print("\nCalculating class centroid...")


centroids = []


for cls in range(6):

    cls_feature = features[
        labels == cls
    ]


    centroid = cls_feature.mean(
        axis=0
    )


    centroids.append(
        centroid
    )


    print(
        class_names[cls],
        cls_feature.shape
    )


centroids = np.array(
    centroids
)


print(
    "Centroid shape:",
    centroids.shape
)


# (6,512)


# =====================================
# 6. Cosine similarity
# =====================================


similarity = cosine_similarity(
    centroids
)


print("\nSimilarity matrix:")

print(similarity)



np.save(
    "features/similarity.npy",
    similarity
)



# =====================================
# 7. Save CSV
# =====================================


import pandas as pd


df = pd.DataFrame(

    similarity,

    index=class_names,

    columns=class_names

)


df.to_csv(
    "features/similarity.csv"
)



print(
    "CSV saved"
)



# =====================================
# 8. Draw heatmap
# =====================================


plt.figure(
    figsize=(7,6)
)


plt.imshow(
    similarity
)


plt.colorbar()



plt.xticks(
    range(6),
    class_names,
    rotation=45
)


plt.yticks(
    range(6),
    class_names
)



plt.title(
    "NTU-Fi HAR Embedding Similarity"
)



for i in range(6):

    for j in range(6):

        plt.text(

            j,

            i,

            f"{similarity[i,j]:.2f}",

            ha="center",

            va="center",

            fontsize=9

        )



plt.tight_layout()



plt.savefig(

    "features/similarity_heatmap.png",

    dpi=300

)



plt.show()



print("\n================================")

print("Embedding Analysis Finished")

print("================================")