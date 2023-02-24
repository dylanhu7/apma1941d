import pickle
import torch

def pickle_to_pt(pickle_file: str, pt_file: str):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    torch.save(data, pt_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle', required=True)
    parser.add_argument('--pt', required=True)
    args = parser.parse_args()
    pickle_to_pt(args.pickle, args.pt)