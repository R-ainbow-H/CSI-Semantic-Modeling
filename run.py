import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from util import load_data_n_model


def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0

    train_loss_history = []
    train_acc_history = []

    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

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
                epoch + 1,
                epoch_accuracy,
                epoch_loss
            )
        )

        # 保存最佳模型
        if epoch_accuracy > best_acc:

            best_acc = epoch_accuracy

            torch.save(
                model.state_dict(),
                "weights/best_resnet18.pth"
            )

            print("Best model updated.")

    # 保存最终模型
    torch.save(
        model.state_dict(),
        "weights/final_resnet18.pth"
    )

    print("\nFinal model saved.")

    np.save("logs/train_loss.npy", np.array(train_loss_history))
    np.save("logs/train_acc.npy", np.array(train_acc_history))

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
            test_loss
        )
    )


def main():

    root = "./Data/"

    parser = argparse.ArgumentParser("WiFi Imaging Benchmark")

    parser.add_argument(
        "--dataset",
        choices=[
            "UT_HAR_data",
            "NTU-Fi-HumanID",
            "NTU-Fi_HAR",
            "Widar",
        ],
    )

    parser.add_argument(
        "--model",
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

    args = parser.parse_args()

    train_loader, test_loader, model, train_epoch = load_data_n_model(
        args.dataset,
        args.model,
        root,
    )

    criterion = nn.CrossEntropyLoss()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    model = train(
        model=model,
        tensor_loader=train_loader,
        num_epochs=train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device,
    )

    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device,
    )


if __name__ == "__ main__":
    main()