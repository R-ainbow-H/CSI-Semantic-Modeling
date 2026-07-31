import argparse
import os


def train(
    model,
    tensor_loader,
    num_epochs,
    learning_rate,
    criterion,
    device,
    model_tag,
    initial_loss_history=None,
    initial_acc_history=None,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_history = list(initial_loss_history or [])
    train_acc_history = list(initial_acc_history or [])
    best_acc = max(train_acc_history) if train_acc_history else 0
    start_epoch = len(train_acc_history)

    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    best_path = os.path.join("weights", f"best_{model_tag}.pth")
    final_path = os.path.join("weights", f"final_{model_tag}.pth")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0

        for inputs, labels in tensor_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1)
            epoch_accuracy += (predict_y == labels).sum().item() / labels.size(0)

        epoch_loss = epoch_loss / len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy / len(tensor_loader)

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_accuracy)

        print(
            "Epoch:{:2d}  Accuracy:{:.4f}  Loss:{:.8f}".format(
                start_epoch + epoch + 1,
                epoch_accuracy,
                epoch_loss,
            )
        )

        if epoch_accuracy > best_acc:
            best_acc = epoch_accuracy
            torch.save(model.state_dict(), best_path)
            print(f"Best model updated: {best_path}")

    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved: {final_path}")

    np.save(os.path.join("logs", f"train_loss_{model_tag}.npy"), np.array(train_loss_history))
    np.save(os.path.join("logs", f"train_acc_{model_tag}.npy"), np.array(train_acc_history))

    return model


def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0

    with torch.no_grad():
        for inputs, labels in tensor_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predict_y = torch.argmax(outputs, dim=1)
            accuracy = (predict_y == labels).sum().item() / labels.size(0)

            test_acc += accuracy
            test_loss += loss.item() * inputs.size(0)

    test_acc = test_acc / len(tensor_loader)
    test_loss = test_loss / len(tensor_loader.dataset)

    print(
        "\nValidation Accuracy:{:.4f}  Loss:{:.6f}".format(
            test_acc,
            test_loss,
        )
    )


def main():
    global np, torch, nn, load_data_n_model

    parser = argparse.ArgumentParser("WiFi Imaging Benchmark")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "UT_HAR_data",
            "NTU-Fi-HumanID",
            "NTU-Fi_HAR",
            "Widar",
        ],
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "MLP",
            "LeNet",
            "ResNet18",
            "ResNet50",
            "ResNet101",
            "RNN",
            "GRU",
            "LSTM",
            "BiLSTM",
            "CNN+GRU",
            "ViT",
        ],
    )
    parser.add_argument("--root", default="./Data/")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the dataset default epoch count.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from weights/final_<dataset>_<model>.pth when present.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Specific checkpoint path to resume from.",
    )
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        import torch.nn as nn

        from util import load_data_n_model
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install PyTorch and run "
            "`pip install -r requirements.txt` in the project environment."
        ) from exc

    train_loader, test_loader, model, train_epoch = load_data_n_model(
        args.dataset,
        args.model,
        args.root,
    )

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_tag = f"{args.dataset}_{args.model}".replace("/", "_").replace("+", "_")

    num_epochs = args.epochs if args.epochs is not None else train_epoch
    initial_loss_history = []
    initial_acc_history = []

    if args.resume or args.resume_checkpoint:
        resume_path = args.resume_checkpoint or os.path.join(
            "weights",
            f"final_{model_tag}.pth",
        )
        if os.path.exists(resume_path):
            model.load_state_dict(torch.load(resume_path, map_location=device))
            print(f"Resumed checkpoint: {resume_path}")
        else:
            print(f"Resume checkpoint not found, training from scratch: {resume_path}")

        loss_path = os.path.join("logs", f"train_loss_{model_tag}.npy")
        acc_path = os.path.join("logs", f"train_acc_{model_tag}.npy")
        if os.path.exists(loss_path) and os.path.exists(acc_path):
            initial_loss_history = np.load(loss_path).tolist()
            initial_acc_history = np.load(acc_path).tolist()
            print(f"Loaded existing logs: {len(initial_acc_history)} epochs")

    model = train(
        model=model,
        tensor_loader=train_loader,
        num_epochs=num_epochs,
        learning_rate=args.learning_rate,
        criterion=criterion,
        device=device,
        model_tag=model_tag,
        initial_loss_history=initial_loss_history,
        initial_acc_history=initial_acc_history,
    )

    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device,
    )


if __name__ == "__main__":
    main()
