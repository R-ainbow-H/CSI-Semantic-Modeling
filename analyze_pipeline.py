import torch
from util import load_data_n_model

# -----------------------------
# Load dataset and model
# -----------------------------
root = "./Data/"

train_loader, test_loader, model, epoch = load_data_n_model(
    "NTU-Fi_HAR",
    "ResNet18",
    root
)

print("=" * 50)
print("SenseFi Pipeline Analysis")
print("=" * 50)

print(f"Train batches : {len(train_loader)}")
print(f"Test batches  : {len(test_loader)}")

print()

# -----------------------------
# Read one sample
# -----------------------------
images, labels = next(iter(train_loader))

print("-" * 50)
print("Input Sample")
print("-" * 50)

print("Input shape :", images.shape)
print("Label shape :", labels.shape)
print("First label :", labels[0].item())

print()

# Only keep one sample
x = images[:1]

print("Single sample shape:", x.shape)

print()

print("-" * 50)
print("Forward Pass")
print("-" * 50)

model.eval()

with torch.no_grad():
    output = model(x)

print()
print("Output logits shape:", output.shape)
print("Predicted class:", torch.argmax(output, dim=1).item())

print()
print("=" * 50)
print("Pipeline Analysis Finished")
print("=" * 50)