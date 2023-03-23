import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
# from scipy.linalg import fractional_matrix_power
import scipy.linalg
import numpy as np

import matplotlib.pyplot as plt

class DLN(nn.Module):
    def __init__(self, d: int, N: int):
        super().__init__()
        self.d = d
        self.N = N
        self.layers = nn.ModuleList(
            [nn.Linear(d, d, bias=False) for _ in range(N)])
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn((d, d))) for _ in range(N)])

    def forward(self) -> Tensor:
        x = torch.eye(self.d)
        for param in self.params:
            x = param @ x
        return x


def upstairs_func(model: DLN,
                  target: Tensor,
                  optimizer: torch.optim.Optimizer,
                  scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None):
    def upstairs(train: bool) -> Tensor:
        model.train(train)
        with torch.set_grad_enabled(train):
            optimizer.zero_grad()
            W = model.forward()
            if train:
                loss = F.mse_loss(torch.diag(W), torch.diag(target))
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                return loss
        return W
    return upstairs

def downstairs_func(model: DLN, target: Tensor, lr: float, N: int):
    def downstairs() -> Tensor:
        model.train(False)
        with torch.no_grad():
            W = model.forward()
            d_F = torch.diag_embed(torch.diag(W - target))
            WWT = W @ W.T
            WTW = W.T @ W
            d_W = torch.zeros_like(W)
            for i in range(1, N + 1):
                # print(d_W)
                # print(torch.pow(WWT, (N - i) / N))
                # print(d_F)
                # print(torch.pow(WTW, (i - 1) / N))
                # print()
                d_W = d_W + \
                    np.array(scipy.linalg.fractional_matrix_power(WWT, (N - i) / N)) @ \
                    np.array(d_F) @ \
                    np.array(scipy.linalg.fractional_matrix_power(WTW, (i - 1) / N))
            d_W = d_W * torch.ones_like(W) / N
            W = W - lr * d_W
            model.params[0] = W
            return F.mse_loss(torch.diag(W), torch.diag(target))
    return downstairs


class DLNArgs(argparse.Namespace):
    d: int
    N: int
    upstairs: bool
    downstairs: bool
    trials: int
    plot: bool
    lr: float


def main(args: DLNArgs):
    d = args.d
    N = args.N
    lr = args.lr
    upstairs_model = DLN(d, N)
    downstairs_model = DLN(d, 1)
    optimizer = torch.optim.Adam(upstairs_model.parameters(), lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)
    target = torch.randn((d, d))
    
    iteration = 0
    upstairs_losses = []
    downstairs_losses = []
    pbar = get_pbar()

    if args.upstairs:
        upstairs = upstairs_func(upstairs_model, target, optimizer)
        while (loss := upstairs(train=True)) > 1e-7:
            upstairs_losses.append(loss.item())
            pbar.update(1)
            pbar.set_postfix_str(f"\b\b\033[1m\033[92mLoss: {loss:.6f}")
            iteration += 1

    if args.downstairs:
        downstairs = downstairs_func(downstairs_model, target, lr, N=3)
        while (loss := downstairs()) > 1e-7:
            downstairs_losses.append(loss.item())
            pbar.update(1)
            pbar.set_postfix_str(f"\b\b\033[1m\033[92mLoss: {loss:.6f}")
            iteration += 1

    pbar.close()
    if args.plot:
        if args.upstairs:
            plot_losses(upstairs_losses, "Upstairs Losses")
        if args.downstairs:
            plot_losses(downstairs_losses, "Upstairs Losses")


def get_pbar():
    return tqdm(total=None, unit=" iterations", desc="\033[1m\033[94mTraining\033[0m", dynamic_ncols=True,
                bar_format="\033[1m{desc}:\033[1m {n_fmt}{unit} | {postfix}\033[0m [{elapsed}, {rate_fmt}]")

def plot_losses(losses: list[float], title: str):
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DLN")
    parser.add_argument('--upstairs', action='store_true')
    parser.add_argument('--downstairs', action='store_true')
    parser.add_argument('--d', type=int, default=2, help="Matrix dimension")
    parser.add_argument('--N', type=int, default=3,
                        help="Number of W matrices upstairs")
    parser.add_argument('--trials', type=int, default=1, help="Number of trials")
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args(namespace=DLNArgs())

    if not (args.upstairs or args.downstairs):
        parser.error("Must specify at least one of --upstairs or --downstairs")

    if args.d < 1:
        parser.error("--d must be a positive integer")

    if args.N < 1:
        parser.error("--N must be a positive integer")

    if args.trials < 1:
        parser.error("--trials must be a positive integer")

    if args.lr < 0:
        parser.error("--lr must be a positive number")

    main(args)
