import os
import argparse
import torch
from torch import Tensor
from typing import Callable
from collections import deque

def text_to_tensor(text: str) -> Tensor:
    idx = lambda c: ord(c) - ord('a') if c != ' ' else 26
    return torch.tensor([idx(c) for c in text])

def mine(corpus: Tensor, corpus_name: str) -> tuple[Tensor, Tensor]:
    counts = torch.ones((27, 27))
    for i in range(len(corpus) - 1):
        counts[corpus[i], corpus[i + 1]] += 1
    P = counts.sum(dim=0) / counts.sum()
    Q = counts / counts.sum(dim=0)
    corpus_name = os.path.basename(corpus_name)
    corpus_name = corpus_name.split('.')[0]
    os.makedirs(f'mined/{corpus_name}')
    torch.save(counts, f'mined/{corpus_name}/tensor.pt')
    torch.save(P, f'mined/{corpus_name}/P.pt')
    torch.save(Q, f'mined/{corpus_name}/Q.pt')
    return P, Q

def energy_func(encoded: Tensor, P: Tensor, Q: Tensor) -> Callable[[Tensor], Tensor]:
    def energy(permutation: Tensor) -> Tensor:
        unpermuted = permutation[encoded]
        return -torch.log(P[unpermuted[0]]) - torch.sum(torch.log(Q[unpermuted[:-1], unpermuted[1:]]))
    return energy

def step_permutation(permutation: Tensor) -> Tensor:
    i, j = torch.randint(0, len(permutation), (2,))
    permutation_copy = permutation.clone()
    permutation_copy[i], permutation_copy[j] = permutation[j], permutation[i]
    return permutation_copy

def stop(prev_energies: deque[Tensor]) -> bool:
    maxlen = prev_energies.maxlen if prev_energies.maxlen is not None else 0
    if len(prev_energies) < maxlen:
        return False
    sum = torch.tensor(0.)
    for i in range(1, len(prev_energies)):
        sum += (prev_energies[i] - prev_energies[i - 1])
    return sum.item() == 0.

def metropolis(encoded: Tensor, P: Tensor, Q: Tensor) -> Tensor:
    energy = energy_func(encoded, P, Q)
    permutation = torch.arange(27)
    count = 0
    prev_energy = energy(permutation)
    prev_energies = deque([prev_energy], maxlen=100)
    while not stop(prev_energies):
        count += 1
        new_permutation = step_permutation(permutation)
        delta_E = (new_energy := energy(new_permutation)) - prev_energy
        if delta_E < 0 or torch.rand(1) < torch.exp(-delta_E):
            decode(new_permutation, encoded)
            prev_energy = new_energy
            prev_energies.append(new_energy.clone())
            permutation = new_permutation
    return permutation
        
def decode(permutation: Tensor, encoded: Tensor) -> str:
    idx = lambda i: chr(i + ord('a')) if i != 26 else ' '
    decoded = ''.join([idx(i) for i in permutation[encoded]])
    with open('decoded.txt', 'w') as f:
        f.write(decoded)
    return decoded

def main(args: argparse.Namespace):
    if args.mined:
        P: Tensor = torch.tensor(torch.load(f'{args.mined}/P.pt'))
        Q: Tensor = torch.tensor(torch.load(f'{args.mined}/Q.pt'))
    else:
        corpus_text = ''
        with open(args.corpus, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    line = line.lower()
                    line = ''.join([c for c in line if c in ' abcdefghijklmnopqrstuvwxyz'])
                    corpus_text += line + ' '
        corpus_text = corpus_text.strip()
        corpus = text_to_tensor(corpus_text)
        P, Q = mine(corpus, args.corpus)

    encoded = ''
    with open(args.input, 'r') as f:
        for line in f:
            encoded += line
    encoded = encoded.strip()
    encoded = text_to_tensor(encoded)

    metropolis(encoded, P, Q)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--corpus', type=str, default=None)
    parser.add_argument('--mined', type=str, default=None)
    args = parser.parse_args()
    if args.corpus is None and args.mined is None:
        raise ValueError('Either --corpus or --mined must be specified.')
    main(args)
