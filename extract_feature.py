import torch
import numpy as np
import os

from util import load_data_n_model


# 保存中间层输出
feature = {}


def hook_fn(module, input, output):
    feature["embedding"] = output.detach()


def main():

    # ======================
    # 1. 加载模型
    # ======================

    root = "./Data/"

    train_loader, test_loader, model, _ = load_data_n_model(
        "NTU-Fi_HAR",
        "ResNet18",
        root
    )


    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    model.eval()


    # ======================
    # 2. 注册hook
    # ======================

    # avgpool输出:
    # [batch,512,1,1]

    model.avgpool.register_forward_hook(
        hook_fn
    )


    # ======================
    # 3. 取一个batch测试
    # ======================

    data_iter = iter(train_loader)

    x, y = next(data_iter)

    x = x.to(device)


    print("Input:")
    print(x.shape)


    with torch.no_grad():

        output = model(x)


    print("\nClassification output:")
    print(output.shape)


    # ======================
    # 4. 获取feature
    # ======================

    emb = feature["embedding"]


    print("\nRaw embedding:")
    print(emb.shape)


    # 去掉 1×1

    emb = emb.squeeze()


    print("\nFinal feature:")
    print(emb.shape)



    # ======================
    # 5. 保存
    # ======================

    os.makedirs(
        "features",
        exist_ok=True
    )


    np.save(
        "features/sample_feature.npy",
        emb.cpu().numpy()
    )


    print("\nSaved:")
    print(
        "features/sample_feature.npy"
    )


if __name__ == "__main__":
    main()