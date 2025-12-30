import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
from datasets.loader import load_dataset
from numpy.random import RandomState
# from visualizer import Visualizer
from metrics import (knn_classify,
                     mean_squared_error,
                     r_squared,
                     affine_align)
import matplotlib.pylab as plt
from models.sde_gpssm import SDE_GPSSM
# from models.sde_gpssm_only_like import SDE_GPSSM

def reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(seed)
    # torch.cuda.manual_seed()为当前GPU设置随机种子
    torch.cuda.manual_seed(seed)

random_seed = 8
reset_seed(random_seed)
device = 'cpu'
rng = np.random.RandomState(random_seed)

# dataset 可选为:
# irregular_toy_example / regular_toy_example
dataset_name = "irregular_toy_example"
epochs = 10000
L_over_2 = 25


def train():
    ds = load_dataset(rng, name=dataset_name)
    Y = ds.Y / 20
    # Y = ds.Y
    # train_Y = Y[:80, :]
    # test_Y = Y[80:, :]
    t = ds.t
    model = SDE_GPSSM(Y, L_over_2, D=ds.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.neg_elbo(Y, t)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1:03d}: Loss = {loss.item():.4f}")

            #### TODO: 这边是 GPLVM 的测试
            # F, K = model.f_eval(batch_y=Y)
            # pred_Y = F.cpu().detach().numpy()
            # mse_Y = mean_squared_error(pred_Y, Y)
            # print(f"\n MSE {mse_Y}")


            if ds.has_true_X:
                # TODO: GPLVM 的测试
                # X_true = ds.X
                # X_pred = model.mu_x.cpu().detach().numpy()
                # X_pred = affine_align(X_pred, X_true=X_true)
                # print(r_squared(X_pred, X_true))

                # TODO: SDE-GPLVM 的测试
                X_true = ds.X
                X_pred, cov_x = model.pred_x(Y, t)
                F, K = model.f_eval(Y, X_pred, cov_x)
                X_pred = X_pred.cpu().detach().numpy()
                print(r_squared(X_pred, X_true))
                X_pred = affine_align(X_pred, X_true=X_true)

                pred_Y = F.cpu().detach().numpy()
                mse_Y = mean_squared_error(pred_Y, Y)
                print(f"\n MSE {mse_Y}")

                plt.figure(figsize=(10, 4))
                plt.plot(pred_Y[:, 1], label='Estimated Y')
                plt.plot(Y[:, 1], label='True Y', linestyle='--')
                plt.xlabel('Time Index')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"figures/epoch_{epoch}_y.png")
                # plt.show()

                # Plot comparison
                plt.figure(figsize=(6, 5))
                plt.plot(X_true[:, 0], X_true[:, 1], label='True X', linewidth=2)
                plt.plot(X_pred[:, 0], X_pred[:, 1], label='Estimated X', linestyle='--', linewidth=2)
                plt.xlabel('X1')
                plt.ylabel('X2')
                plt.grid(True)
                plt.axis('equal')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"figures/epoch_{epoch}_x.png")
                # plt.show()

            # print(f"\n MSE {mse_Y}")

            if ds.is_categorical:
                knn_acc = knn_classify(model.mu_x.cpu().detach().numpy(), ds.labels, rng)
                print(f"\n KNN acc {knn_acc:04f}")

    print("Training complete.")


if __name__ == "__main__":
    train()