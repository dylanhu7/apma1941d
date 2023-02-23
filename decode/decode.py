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
    permutation[i], permutation[j] = permutation_copy[j], permutation_copy[i]
    return permutation.clone()

def metropolis(encoded: Tensor, P: Tensor, Q: Tensor) -> Tensor:
    energy = energy_func(encoded, P, Q)
    permutation = torch.arange(27)
    count = 0
    prev = energy(permutation)
    while True:
        count += 1
        new_permutation = step_permutation(permutation)
        delta_E = (new_energy := energy(new_permutation)) - prev
        if delta_E < 0 or torch.rand(1) < torch.exp(-delta_E):
            print(count, new_energy)
            decode(new_permutation, encoded)
            prev = new_energy
            permutation = new_permutation
        
def decode(permutation: Tensor, encoded: Tensor) -> str:
    idx = lambda i: chr(i + ord('a')) if i != 26 else ' '
    decoded = ''.join([idx(i) for i in permutation[encoded]])
    with open('decoded.txt', 'w') as f:
        f.write(decoded)
    return decoded

def decode_test():
    identity = torch.arange(27)
    encoded = text_to_tensor('hello world')
    assert decode(identity, encoded) == 'hello world'
    permutation = step_permutation(identity)
    print(decode(permutation, encoded))

def main(args: argparse.Namespace):
    if args.corpus is None and args.mined is None:
        raise ValueError('Either --corpus or --mined must be specified.')

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

    # decode_test()
    metropolis(encoded, P, Q)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--corpus', type=str, default=None)
    parser.add_argument('--mined', type=str, default=None)
    args = parser.parse_args()
    main(args)


