import click
import torch
import torch.nn as nn

class MultiplicativeBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = nn.Parameter(torch.tensor([1.0], dtype=torch.float))

    def forward(self, x):
        return x * self.k

def train_buffer(buffer, estimated_distance, true_distances, epochs = 100, lr = 0.01):
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(estimated_distance, true_distances),
        batch_size=len(estimated_distance),
        shuffle=True
    )

    optimizer = torch.optim.Adam(buffer.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in dataloader:
            batch_estimated_distances, batch_true_distances = batch
            optimizer.zero_grad()
            loss = (buffer(batch_estimated_distances) - batch_true_distances) ** 2
            loss.backward()
            optimizer.step()

def evaluate_buffer(buffer, estimated_distances, true_distances):
    predicted_true_distances = buffer(estimated_distances)
    return torch.mean((predicted_true_distances - true_distances) ** 2)
    # Potrebbe servire anche un grafico per vedere la distribuzione delle distanze 

# Si può fare uno studio sulle curve ROC? Se sì, mi servono anche i genuine

@click.command()
@click.argument('domain')
@click.argument('architecture')
@click.argument('test')
def main(domain, architecture, test):
    pass

if __name__ == '__main__':
    main()