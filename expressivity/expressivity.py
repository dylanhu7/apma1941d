import torch
import csv
from torch import nn
from torch import Tensor

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
    
def train(model: MultiLayerPerceptron, x: Tensor, y: Tensor, epochs: int, learning_rate: float):
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_func(outputs, y)

        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

def load_data(file_path) -> list[tuple[Tensor, Tensor]]:
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            # Convert each row to a list of floats and then to a tensor
            y = torch.tensor([float(value) for value in row], dtype=torch.float32)[..., None]
            x = torch.linspace(0, 1, y.shape[0])[..., None]
            data.append((x, y))
    return data


def main():
    data = load_data('data.csv')

    model = MultiLayerPerceptron([1, 5, 3, 5, 1], activation_fn=nn.ELU())
    print(model)

    # Train the model
    for x, y in data:
        train(model, x, y, epochs=1000, learning_rate=0.001)

if __name__ == '__main__':
    main()