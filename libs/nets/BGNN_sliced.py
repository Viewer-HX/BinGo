import torch
from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class BGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(BGNN, self).__init__()
        torch.manual_seed(23456)
        self.conv1A = GCNConv(num_node_features, 32)
        self.conv1B = GCNConv(num_node_features, 32)
        self.conv1C = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 8)
        self.mlp = Sequential(
            Linear(16, 8),
            ReLU(),
            Linear(8, 2)
        )

    def forward(self, x0, edge_attr0, edge_index0, x1, edge_attr1, edge_index1, batch0, batch1):
        # 1. Obtain node embeddings
        x0A = self.conv1A(x0, edge_index0, edge_attr0[:, 0].contiguous()).relu()
        x0B = self.conv1B(x0, edge_index0, edge_attr0[:, 1].contiguous()).relu()
        x0C = self.conv1C(x0, edge_index0, edge_attr0[:, 2].contiguous()).relu()
        x0 = (x0A + x0B + x0C) / 3.0
        x0 = self.conv2(x0, edge_index0)
        x0 = x0.relu()
        x0 = self.conv3(x0, edge_index0)
        # 2. Obtain node embeddings
        x1A = self.conv1A(x1, edge_index1, edge_attr1[:, 0].contiguous()).relu()
        x1B = self.conv1B(x1, edge_index1, edge_attr1[:, 1].contiguous()).relu()
        x1C = self.conv1C(x1, edge_index1, edge_attr1[:, 2].contiguous()).relu()
        x1 = (x1A + x1B + x1C) / 3.0
        x1 = self.conv2(x1, edge_index1)
        x1 = x1.relu()
        x1 = self.conv3(x1, edge_index1)
        # 3. Readout layer
        x0 = global_mean_pool(x0, batch0)  # [batch_size, hidden_channels]
        x1 = global_mean_pool(x1, batch1)  # [batch_size, hidden_channels]
        # 4. Concatenate
        x = torch.cat([x0, x1], dim=1)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.mlp(x)

        return x

def BGNNTrain(model, trainloader, optimizer, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    lossTrain = 0
    for data in trainloader:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()  # Clear gradients.
        data.to(device)
        out = model.forward(data.x_s, data.edge_attr_s, data.edge_index_s, data.x_t, data.edge_attr_t, data.edge_index_t, data.x_s_batch, data.x_t_batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        # statistic.
        lossTrain += loss.item() * len(data.y)

    lossTrain /= len(trainloader.dataset)

    return model, lossTrain

def BGNNTest(model, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    preds = []
    labels = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        out = model.forward(data.x_s, data.edge_attr_s, data.edge_index_s, data.x_t, data.edge_attr_t, data.edge_index_t, data.x_s_batch, data.x_t_batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        # statistic.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        preds.extend(pred.int().tolist())
        labels.extend(data.y.int().tolist())

    acc = correct / len(loader.dataset)  # Derive ratio of correct predictions.

    return acc, preds, labels
