import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

import matplotlib.pyplot as plt

class DLN(nn.Module):
    def __init__(self, d: int, N: int):
        super().__init__()
        self.d = d
        self.N = N
        self.layers = nn.ModuleList(
            [nn.Linear(d, d, bias=False) for _ in range(N)])
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight)

    def forward(self) -> Tensor:
        x = torch.eye(self.d)
        for layer in self.layers:
            x = layer(x)
        return x


def upstairs_func(model: DLN,
                  optimizer: torch.optim.Optimizer,
                  scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None):
    def upstairs(target: Tensor, train: bool) -> Tensor:
        model.train(train)
        with torch.set_grad_enabled(train):
            optimizer.zero_grad()
            output = model()
            if train:
                loss = F.mse_loss(torch.diag(output), torch.diag(target))
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                return loss
        return output
    return upstairs


class DLNArgs(argparse.Namespace):
    d: int
    N: int
    upstairs: bool
    downstairs: bool
    trials: int
    plot: bool


def main(args: DLNArgs):
    d = args.d
    N = args.N
    model = DLN(d, N)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)
    upstairs = upstairs_func(model, optimizer)
    target = torch.randn((d, d))
    iteration = 0
    losses = []
    pbar = get_pbar()

    if args.upstairs:
        loss = upstairs(target, True)
        while (loss := upstairs(target, True)) > 1e-7:
            losses.append(loss.item())
            pbar.update(1)
            pbar.set_postfix_str(f"\b\b\033[1m\033[92mLoss: {loss:.6f}")
            iteration += 1

    pbar.close()
    if args.plot:
        plot_losses(losses, "Upstairs Losses")
    # print(target)
    # print(upstairs(target, False))


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
    args = parser.parse_args(namespace=DLNArgs())

    if not (args.upstairs or args.downstairs):
        parser.error("Must specify at least one of --upstairs or --downstairs")

    if args.d < 1:
        parser.error("--d must be a positive integer")

    if args.N < 1:
        parser.error("--N must be a positive integer")

    main(args)
