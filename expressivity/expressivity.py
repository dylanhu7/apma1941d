import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
from torch import Tensor, nn


class Args(argparse.Namespace):
    depth: int
    width: int
    use_positional_encoding: bool
    pe_mode: str
    pe_num_bands: int
    output_dir: str
    activation_fn: str
    lr: float


class MultiLayerPerceptron(nn.Module):
    def __init__(self, dimensions: list[int], activation_fn: nn.Module = nn.ReLU()):
        super(MultiLayerPerceptron, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(dimensions)-2): # stop one step earlier
            self.layers.append(nn.Linear(dimensions[i], dimensions[i+1]))
            self.layers.append(activation_fn)

        # add the last layer without activation function
        self.layers.append(nn.Linear(dimensions[-2], dimensions[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class PositionalEncoding(nn.Module):
    """
    Positional encoding module
    """
    def __init__(self, mode: str, num_bands: int):
        super(PositionalEncoding, self).__init__()
        self.mode = mode
        self.num_bands = num_bands

    def sincos_encoding(self, x):
        """
        Encode the input tensor using the sine and cosine functions
        """
        encoding_fns = [lambda x: x]
        for i in range(self.num_bands):
            encoding_fns.append(
                lambda x, i=i: torch.sin(2 ** i * x))
            encoding_fns.append(
                lambda x, i=i: torch.cos(2 ** i * x))
        return torch.cat([fn(x) for fn in encoding_fns], dim=-1)
    
    def fourier_encoding(self, x):
        """
        Encode the input tensor using random Fourier features
        """
        for i in range(self.num_bands):
            x = torch.cat((torch.sin((2**i)*x), torch.cos((2**i)*x)), dim=-1)
        return x

    def forward(self, x):
        if self.mode == 'sincos':
            return self.sincos_encoding(x)
        elif self.mode == 'fourier':
            return self.fourier_encoding(x)
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
    
def train(model, x, y, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    predictions = []
    losses = []

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        losses.append(loss.item())
        predictions.append(outputs.detach().numpy())

        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
            
    return predictions, losses



def load_data(file_path) -> list[tuple[Tensor, Tensor]]:
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        x = torch.tensor([float(value) for value in next(reader)], dtype=torch.float32)[..., None]
        for row in reader:
            y = torch.tensor([float(value) for value in row], dtype=torch.float32)[..., None]
            data.append((x, y))
    return data

def plot_losses(losses: list[float]):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

def animate_training(x: Tensor, ground_truth: Tensor, predictions: list[Tensor], losses: list[float], args: Args):
    fig, ax = plt.subplots()
    line1, = ax.plot(x, ground_truth.detach().numpy(), label='Ground truth')  # Plot the true values
    line2, = ax.plot(x, predictions[0], label='Model approximation')  # Plot the model's initial predictions
    plt.legend()

    epoch_text = ax.text(0.02, 0.8, '', transform=ax.transAxes, color='black')
    depth_text = ax.text(0.02, 0.75, f'Depth: {args.depth}', transform=ax.transAxes, color='gray')
    width_text = ax.text(0.02, 0.7, f'Width: {args.width}', transform=ax.transAxes, color='gray')
    activation_fn_text = ax.text(0.02, 0.65, f'Width: {args.width}', transform=ax.transAxes, color='gray')
    lr_text = ax.text(0.02, 0.6, f'Learning rate: {args.activation_fn}', transform=ax.transAxes, color='gray')
    num_bands_text = ax.text(0.02, 0.55, f'Number of bands: {args.pe_num_bands}', transform=ax.transAxes, color='gray') if args.use_positional_encoding else None

    def update(num):
        line2.set_ydata(predictions[num])  # Update the model's predictions
        epoch_text.set_text(f'Epoch: {num + 1}')  # Update the epoch counter
        depth_text.set_text(f'Depth: {args.depth}')
        width_text.set_text(f'Width: {args.width}')
        activation_fn_text.set_text(f'Activation function: {args.activation_fn}')
        lr_text.set_text(f'Learning rate: {args.lr}')
        if num_bands_text is not None:
            num_bands_text.set_text(f'Number of bands: {args.pe_num_bands}')
        return line2,

    animation = FuncAnimation(fig, update, frames=range(len(predictions)), blit=True)
    plt.close(fig)
    return animation

def get_filename(args: Args, i: int):
    name_components: list[str] = []
    name_components.append("d" + str(args.depth))
    name_components.append("w" + str(args.width))
    if args.use_positional_encoding:
        name_components.append(args.pe_mode)
        name_components.append("b" + str(args.pe_num_bands))
    name_components.append(str(i))
    return "_".join(name_components)

def write_animation(animation: FuncAnimation, args: Args, i: int):
    filename = get_filename(args, i)
    outfile = os.path.join(args.output_dir, f"{filename}.mp4")
    animation.save(outfile, writer='ffmpeg', fps=50)

def activation_fn(name: str):
    if name == 'none':
        return nn.Identity()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f'Unknown activation function: {name}')


def main(args: Args):
    data = load_data('data.csv')

    encoder = PositionalEncoding(mode='sincos', num_bands=args.pe_num_bands)

    # Train the model
    for i, (x, y) in enumerate(data):
        if args.use_positional_encoding:
            x_enc = encoder(x)
            input_dim = x_enc.shape[-1]
            model = MultiLayerPerceptron([input_dim, 16, 16, 16, 1], activation_fn=activation_fn(args.activation_fn))
            predictions, losses = train(model, x_enc, y, epochs=1000, learning_rate=args.lr)
        else:
            model = MultiLayerPerceptron([1, 16, 16, 16, 1], activation_fn=nn.ELU())
            predictions, losses = train(model, x, y, epochs=1000, learning_rate=0.001)
        write_animation(animate_training(x, y, predictions, losses, args), args, i)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--use_positional_encoding', action='store_true', default=False)
    parser.add_argument('--pe_mode', type=str, default='sincos')
    parser.add_argument('--pe_num_bands', type=int, default=6)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--activation_fn', type=str, default='elu', choices=['elu', 'relu', 'sigmoid', 'tanh', 'leakyrelu', 'none'])
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args(namespace=Args())
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)