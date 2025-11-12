import torch
import matplotlib.pyplot as plt
import seaborn as sns

@torch.no_grad()
def analyse_tensor(tensor : torch.tensor):
    tensor = tensor.reshape(-1, tensor.shape[-1]).clone().cpu()
    tmp_acti = tensor.abs().to(torch.float)
    tmp_array = tmp_acti.numpy()
    plt.figure()
    sns.heatmap(
        tmp_array.T,  # 转置使特征维度在y轴
        cmap="viridis",
        cbar_kws={"label": "Activation"},
        xticklabels=(tensor.shape[0] + 31) // 32,
        yticklabels=(tensor.shape[1] + 31) // 32
    )
    plt.title("Input Activations Across Sequence")
    plt.xlabel("Token Position")
    plt.ylabel("Feature Dimension")
    plt.tight_layout()

    tensor = tensor.abs()
    average = tensor.mean()
    maxItem = tensor.max()
    print("average:", average)
    print("maxItem:", maxItem)
    # print(tensor.shape)
    # print(tensor.numel())
    # print(tensor.numel() * 0.01)
    # # idx = torch.nonzero(tensor.abs() > (average + maxItem)/10)
    # idx = torch.topk(tensor, round(tensor.numel() * 0.01), largest=True).indices
    # print(idx.shape)
    # n, _ = idx.shape
    # print("Outlier Rate:", n/tensor.numel())
    # rows = idx[:,0]
    # cols = idx[:,1]
    # rowCounts = torch.bincount(rows)
    # colCounts = torch.bincount(cols)
    # plt.figure()
    # plt.plot(rowCounts, color = 'r')
    # plt.title(f"Outlier Distribute")
    # plt.xlabel("Input rows")
    # plt.ylabel("Frequancy")
    # plt.grid(True)

    # plt.figure()
    # plt.plot(colCounts, color = 'b')
    # plt.title(f"Outlier Distribute")
    # plt.xlabel("Input cols")
    # plt.ylabel("Frequancy")
    # plt.grid(True)
    plt.show()
    # print(rowCounts)
    # print(colCounts)

@torch.no_grad()
def analyse_tensor_dim(tensor : torch.tensor):
    tensor = tensor.reshape(-1, tensor.shape[-1])
    tensor = tensor.abs()
    average = tensor.mean()
    maxItem = tensor.max()
    print("average:", average)
    print("maxItem:", maxItem)
    # print(tensor.shape)
    # print(tensor.numel())
    # print(tensor.numel() * 0.01)
    # # idx = torch.nonzero(tensor.abs() > (average + maxItem)/10)
    # idx = torch.topk(tensor, round(tensor.numel() * 0.01), largest=True).indices
    # print(idx.shape)
    # n, _ = idx.shape
    # print("Outlier Rate:", n/tensor.numel())
    # rows = idx[:,0]
    # cols = idx[:,1]
    # rowCounts = torch.bincount(rows)
    # colCounts = torch.bincount(cols)
    # plt.figure()
    # plt.plot(rowCounts, color = 'r')
    # plt.title(f"Outlier Distribute")
    # plt.xlabel("Input rows")
    # plt.ylabel("Frequancy")
    # plt.grid(True)

    # plt.figure()
    # plt.plot(colCounts, color = 'b')
    # plt.title(f"Outlier Distribute")
    # plt.xlabel("Input cols")
    # plt.ylabel("Frequancy")
    # plt.grid(True)
    plt.show()
    # print(rowCounts)
    # print(colCounts)

if __name__ == "__main__":
    t = torch.randn(512, 3584)
    analyse_tensor(t)