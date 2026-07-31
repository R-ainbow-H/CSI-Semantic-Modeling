import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")


DEFAULT_CLASS_NAMES = {
    "UT_HAR_data": [
        "lie_down",
        "fall",
        "walk",
        "pickup",
        "run",
        "sit_down",
        "stand_up",
    ],
    "NTU-Fi_HAR": [
        "box",
        "circle",
        "clean",
        "fall",
        "run",
        "walk",
    ],
    "NTU-Fi-HumanID": [
        "001",
        "002",
        "003",
        "004",
        "005",
        "006",
        "007",
        "008",
        "009",
        "010",
        "011",
        "012",
        "013",
        "015",
    ],
}


WIDAR_CLASS_NAMES = sorted([
    "1-Push&Pull",
    "2-Sweep",
    "3-Clap",
    "4-Slide",
    "5-Draw-N(H)",
    "6-Draw-O(H)",
    "7-Draw-Rectangle(H)",
    "8-Draw-Triangle(H)",
    "9-Draw-Zigzag(H)",
    "10-Draw-Zigzag(V)",
    "11-Draw-N(V)",
    "12-Draw-O(V)",
    "13-Draw-1",
    "14-Draw-2",
    "15-Draw-3",
    "16-Draw-4",
    "17-Draw-5",
    "18-Draw-6",
    "19-Draw-7",
    "20-Draw-8",
    "21-Draw-9",
    "22-Draw-10",
])


def parse_args():
    parser = argparse.ArgumentParser("CSI embedding analysis")
    parser.add_argument("--dataset", default="NTU-Fi_HAR")
    parser.add_argument("--model", default="ResNet18")
    parser.add_argument("--root", default="./Data/")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save-dir", default="features")
    parser.add_argument("--batch-limit", type=int, default=0)
    parser.add_argument("--analysis-sample-size", type=int, default=5000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Reuse features.npy and labels.npy from --save-dir when present.",
    )
    return parser.parse_args()


def safe_tag(dataset_name, model_name):
    return f"{dataset_name}_{model_name}".replace("/", "_").replace("+", "_")


def default_checkpoint_path(dataset_name, model_name):
    return Path("weights") / f"best_{safe_tag(dataset_name, model_name)}.pth"


def resolve_checkpoint_path(dataset_name, model_name, explicit_checkpoint):
    if explicit_checkpoint:
        return Path(explicit_checkpoint)

    candidates = [
        default_checkpoint_path(dataset_name, model_name),
        Path("weights") / f"final_{safe_tag(dataset_name, model_name)}.pth",
    ]

    if model_name == "ResNet18":
        candidates.extend(
            [
                Path("weights") / "best_resnet18.pth",
                Path("weights") / "final_resnet18.pth",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def infer_class_names(dataset_name, loader):
    dataset = loader.dataset
    if hasattr(dataset, "category"):
        return [name for name, _ in sorted(dataset.category.items(), key=lambda item: item[1])]

    if dataset_name == "Widar":
        return WIDAR_CLASS_NAMES

    return DEFAULT_CLASS_NAMES.get(dataset_name)


def build_resnet_feature_extractor(model):
    required_layers = [
        "reshape",
        "conv1",
        "batch_norm1",
        "relu",
        "max_pool",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "avgpool",
    ]
    missing = [name for name in required_layers if not hasattr(model, name)]
    if missing:
        raise ValueError(
            "Embedding analysis currently supports ResNet-style SenseFi models. "
            f"Missing layers: {', '.join(missing)}"
        )

    return nn.Sequential(*(getattr(model, name) for name in required_layers))


def take_analysis_sample(features, labels, sample_size, random_state=42):
    if sample_size <= 0 or len(features) <= sample_size:
        return features, labels

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(features), size=sample_size, replace=False)
    return features[indices], labels[indices]


def save_matrix_csv(path, matrix, class_names):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class"] + class_names)
        for name, row in zip(class_names, matrix):
            writer.writerow([name] + [float(value) for value in row])


def plot_embedding(points, labels, class_names, title, path):
    plt.figure(figsize=(8, 6))

    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if not np.any(mask):
            continue
        plt.scatter(
            points[mask, 0],
            points[mask, 1],
            s=15,
            label=class_name,
            alpha=0.75,
        )

    plt.legend(fontsize=8)
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_heatmap(matrix, class_names, title, path):
    plt.figure(figsize=(max(7, len(class_names) * 0.42), max(6, len(class_names) * 0.38)))
    plt.imshow(matrix)
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(class_names)), class_names, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    global plt, np, torch, nn, cdist, KMeans, PCA, TSNE
    global adjusted_rand_score, silhouette_score, load_data_n_model

    args = parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        import torch.nn as nn
        from scipy.spatial.distance import cdist
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.metrics import adjusted_rand_score, silhouette_score

        from util import load_data_n_model
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install PyTorch and run "
            "`pip install -r requirements.txt` in the project environment."
        ) from exc

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    features_path = save_dir / "features.npy"
    labels_path = save_dir / "labels.npy"

    if args.reuse_features and features_path.exists() and labels_path.exists():
        print("=" * 60)
        print("Reusing saved features")
        print("=" * 60)

        features = np.load(features_path)
        labels = np.load(labels_path).astype(int)
        class_names = DEFAULT_CLASS_NAMES.get(args.dataset)
        if args.dataset == "Widar":
            class_names = WIDAR_CLASS_NAMES
        checkpoint_path = resolve_checkpoint_path(args.dataset, args.model, args.checkpoint)
    else:
        print("=" * 60)
        print("Loading dataset and model")
        print("=" * 60)

        train_loader, test_loader, model, _ = load_data_n_model(
            args.dataset,
            args.model,
            args.root,
        )
        loader = train_loader if args.split == "train" else test_loader

        class_names = infer_class_names(args.dataset, loader)
        if not class_names:
            raise ValueError(f"Cannot infer class names for dataset: {args.dataset}")

        checkpoint_path = resolve_checkpoint_path(
            args.dataset,
            args.model,
            args.checkpoint,
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Train first with run.py or pass --checkpoint explicitly."
            )

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        feature_extractor = build_resnet_feature_extractor(model).to(device)
        feature_extractor.eval()

        print("Feature extractor ready")
        print("\nExtracting features...")

        features = []
        labels = []

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(loader):
                if args.batch_limit and batch_idx >= args.batch_limit:
                    break

                x = x.to(device)
                embedding = feature_extractor(x)
                embedding = embedding.reshape(embedding.shape[0], -1)

                features.append(embedding.cpu().numpy())
                labels.append(y.cpu().numpy())

                print(f"Batch {batch_idx + 1}/{len(loader)} finished")

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0).astype(int)

        np.save(features_path, features)
        np.save(labels_path, labels)

    if not class_names:
        raise ValueError(f"Cannot infer class names for dataset: {args.dataset}")

    print("\nFeature extraction finished")
    print("Feature shape:", features.shape)
    print("Label shape:", labels.shape)

    analysis_features, analysis_labels = take_analysis_sample(
        features,
        labels,
        args.analysis_sample_size,
    )
    sample_note = (
        f"{len(analysis_features)} sampled points"
        if len(analysis_features) != len(features)
        else "all points"
    )

    metrics = {
        "dataset": args.dataset,
        "model": args.model,
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "feature_shape": list(features.shape),
        "label_shape": list(labels.shape),
        "analysis_points": int(len(analysis_features)),
        "feature_mean": float(np.mean(features)),
        "feature_std": float(np.std(features)),
        "feature_min": float(np.min(features)),
        "feature_max": float(np.max(features)),
    }

    print("\n" + "=" * 60)
    print("Cluster Evaluation")
    print("=" * 60)

    metrics["silhouette_score"] = float(
        silhouette_score(analysis_features, analysis_labels)
    )
    print(f"Silhouette Score ({sample_note}): {metrics['silhouette_score']}")

    kmeans = KMeans(
        n_clusters=len(class_names),
        random_state=42,
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(analysis_features)
    metrics["adjusted_rand_score"] = float(
        adjusted_rand_score(analysis_labels, cluster_labels)
    )
    np.save(save_dir / "kmeans_labels.npy", cluster_labels)
    print(f"ARI ({sample_note}): {metrics['adjusted_rand_score']}")

    print("\nCalculating centroids...")
    centroids = []
    intra_distance = {}

    for class_idx, class_name in enumerate(class_names):
        class_features = features[labels == class_idx]
        if len(class_features) == 0:
            raise ValueError(f"No features found for class {class_idx}: {class_name}")

        centroid = np.mean(class_features, axis=0)
        centroids.append(centroid)

        distance = np.mean(np.linalg.norm(class_features - centroid, axis=1))
        intra_distance[class_name] = float(distance)
        print(f"{class_name}: count={len(class_features)}, intra_distance={distance:.6f}")

    centroids = np.array(centroids)

    norm = np.linalg.norm(centroids, axis=1, keepdims=True)
    normalized_centroids = centroids / np.maximum(norm, 1e-12)
    similarity_matrix = normalized_centroids @ normalized_centroids.T
    distance_matrix = cdist(centroids, centroids, metric="euclidean")

    save_matrix_csv(save_dir / "similarity.csv", similarity_matrix, class_names)
    save_matrix_csv(save_dir / "distance_matrix.csv", distance_matrix, class_names)
    plot_heatmap(
        similarity_matrix,
        class_names,
        "Cosine Similarity Matrix",
        save_dir / "similarity_heatmap.png",
    )
    plot_heatmap(
        distance_matrix,
        class_names,
        "Euclidean Distance Matrix",
        save_dir / "distance_heatmap.png",
    )

    print("\nRunning PCA...")
    pca = PCA(n_components=2)
    feature_pca = pca.fit_transform(features)
    plot_embedding(feature_pca, labels, class_names, "PCA of CSI Embedding", save_dir / "pca.png")

    print("Running t-SNE...")
    tsne_features, tsne_labels = take_analysis_sample(
        features,
        labels,
        args.analysis_sample_size,
    )
    perplexity = min(args.tsne_perplexity, max(5.0, (len(tsne_features) - 1) / 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    feature_tsne = tsne.fit_transform(tsne_features)
    plot_embedding(
        feature_tsne,
        tsne_labels,
        class_names,
        f"t-SNE of CSI Embedding ({len(tsne_features)} points)",
        save_dir / "tsne.png",
    )

    metrics["intra_class_distance"] = intra_distance
    metrics["class_names"] = class_names
    metrics["tsne_points"] = int(len(tsne_features))
    metrics["tsne_perplexity"] = float(perplexity)

    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(save_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("CSI Embedding Analysis Report\n")
        f.write("=" * 32 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Feature shape: {features.shape}\n")
        f.write(f"Analysis points: {len(analysis_features)}\n\n")
        f.write(f"Silhouette Score: {metrics['silhouette_score']}\n")
        f.write(f"Adjusted Rand Index: {metrics['adjusted_rand_score']}\n\n")
        f.write("Intra-class distance:\n")
        for class_name, distance in intra_distance.items():
            f.write(f"- {class_name}: {distance}\n")
        f.write("\nGenerated files:\n")
        for filename in [
            "features.npy",
            "labels.npy",
            "pca.png",
            "tsne.png",
            "similarity.csv",
            "distance_matrix.csv",
            "similarity_heatmap.png",
            "distance_heatmap.png",
            "kmeans_labels.npy",
            "metrics.json",
        ]:
            f.write(f"- {filename}\n")

    print("\n" + "=" * 60)
    print("Analysis Finished")
    print("=" * 60)
    print("Results saved in:", save_dir)


if __name__ == "__main__":
    main()
