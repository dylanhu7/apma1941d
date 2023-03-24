import argparse
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# from scipy.linalg import fractional_matrix_power
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm


class DLN(nn.Module):
    def __init__(self, d: int, N: int):
        super().__init__()
        self.d = d
        self.N = N
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
                d_W = d_W + \
                    scipy.linalg.fractional_matrix_power(
                        WWT, (N - i) / N) @ \
                    np.array(d_F) @ \
                    scipy.linalg.fractional_matrix_power(
                        WTW, (i - 1) / N)
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
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)
    target = torch.randn((d, d))

    iteration = 0
    upstairs_losses: dict[int, list[float]] = {}
    upstairs_losses_adam: dict[int, list[float]] = {}
    upstairs_losses_rms: dict[int, list[float]] = {}
    downstairs_losses: dict[int, list[float]] = {}
    pbar = get_pbar()

    for trial in range(args.trials):
        upstairs_losses[trial] = []
        downstairs_losses[trial] = []

        if args.upstairs:
            upstairs_model = DLN(d, N)
            optimizer = torch.optim.SGD(upstairs_model.parameters(), lr)
            upstairs = upstairs_func(upstairs_model, target, optimizer)
            while (loss := upstairs(train=True).item()) > 1e-7:
                upstairs_losses[trial].append(loss)
                pbar.update(1)
                pbar.set_postfix_str(f"\b\b\033[1m\033[92mLoss: {loss:.6f}")
                iteration += 1

        if args.downstairs:
            downstairs_model = DLN(d, 1)
            downstairs = downstairs_func(downstairs_model, target, lr, N=3)
            while (loss := downstairs().item()) > 1e-7:
                downstairs_losses[trial].append(loss)
                pbar.update(1)
                pbar.set_postfix_str(f"\b\b\033[1m\033[92mLoss: {loss:.6f}")
                iteration += 1

    pbar.close()
    if args.plot:
            plot_losses([upstairs_losses, downstairs_losses], "Upstairs vs Downstairs | d = 2, N = 3 | lr = 0.01")


def get_pbar():
    return tqdm(total=None, unit=" iterations", desc="\033[1m\033[94mTraining\033[0m", dynamic_ncols=True,
                bar_format="\033[1m{desc}:\033[1m {n_fmt}{unit} | {postfix}\033[0m [{elapsed}, {rate_fmt}]")


def plot_losses(variants: list[dict[int, list[float]]], title: str):
    i = 0
    for variant in variants:
        avg_trajectory = np.zeros(max(len(losses) for losses in variant.values()))
        iters = []
        for trial, losses in variant.items():
            avg_trajectory[:len(losses)] += np.array(losses)
            iters.append(len(losses))
        mean = np.mean(iters)
        std = np.std(iters)
        print(f"Mean: {mean:.6f} | Std: {std:.6f}")
        avg_trajectory /= len(variant)
        plt.plot(avg_trajectory, label="Upstairs" if i == 0 else "Downstairs")
        i += 1
    plt.xlabel("Iteration")
    plt.xlim(0, 500)
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DLN")
    parser.add_argument('--upstairs', action='store_true')
    parser.add_argument('--downstairs', action='store_true')
    parser.add_argument('--d', type=int, default=2, help="Matrix dimension")
    parser.add_argument('--N', type=int, default=3,
                        help="Number of W matrices upstairs")
    parser.add_argument('--trials', type=int, default=1,
                        help="Number of trials")
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
