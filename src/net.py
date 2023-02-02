import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, n_filters, n_layers):
        super(FeatureExtractor, self).__init__()
        self.convs = []
        self.bns = []
        self.n_layers = n_layers
        self.n_filters = n_filters

        for i in range(n_layers):
            self.convs.append(
                nn.Conv2d(
                    in_channels if i == 0 else n_filters,
                    n_filters,
                    3,
                    padding=1,
                    bias=False
                )
            )
            self.bns.append(nn.BatchNorm2d(n_filters))
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = self.relu(x)
        return x


class GoNetwork(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor, out_size, activation_function):
        super(GoNetwork, self).__init__()
        self.feature_extractor = feature_extractor

        self.conv = nn.Conv2d(self.feature_extractor.n_filters, 1, 1)
        self.bn = nn.BatchNorm2d(1)

        self.linear = nn.Linear(9 * 9, out_size)

        self.activation_function = activation_function

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.conv(x)
        x = self.bn(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        x = self.activation_function(x)
        return x


def train(
        network,
        train_dataset,
        test_dataset,
        target_name,
        criterion,
        optimizer,
        predict_f,
        writer,
        epochs=100,
        batch_size=512
):
    train_dataset = TensorDataset(train_dataset["X"], train_dataset[target_name])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)

    test_dataset = TensorDataset(test_dataset["X"], test_dataset[target_name])
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    for step in tqdm(range(epochs), desc="Step training"):
        running_loss = 0
        for inputs, targets in tqdm(train_loader, desc="Batch training"):
            outputs = network(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        writer.add_scalar(f"Loss/{target_name}", running_loss / len(train_loader), step)

        if step % 10 == 0:
            running_accuracy = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = network(inputs)
                    running_accuracy += torch.count_nonzero(predict_f(outputs) == predict_f(targets)) / targets.shape[0]

            writer.add_scalar(f"Accuracy/{target_name}", running_accuracy / len(test_loader), step)


if __name__ == "__main__":
    writer = SummaryWriter
    dataset = torch.load("dataset.pt")
    train_dataset = {}
    test_dataset = {}
    n = dataset["X"].shape[0]
    indices = torch.randperm(n)
    train_indices = indices[:int(0.75 * n)]
    test_indices = indices[int(0.75 * n):]

    for k, v in dataset.items():
        train_dataset[k] = v[train_indices]
        test_dataset[k] = v[test_indices]

    feature_network1 = FeatureExtractor(dataset["X"].shape[1], 128, 11)
    policy_network = GoNetwork(feature_network1, 81, nn.functional.softmax)

    print("Start policy network training...")
    train(
        policy_network,
        train_dataset,
        test_dataset,
        "policy_data",
        nn.CrossEntropyLoss(),
        torch.optim.Adam(policy_network.parameters(), lr=0.001),
        lambda x: x.argmax(-1),
        writer
    )

    feature_network2 = FeatureExtractor(dataset["X"].shape[1], 128, 11)
    feature_network2.load_state_dict(feature_network1.state_dict())
    value_network = GoNetwork(feature_network2, 1, nn.functional.sigmoid)

    print("Start value network training...")
    train(
        value_network,
        train_dataset,
        test_dataset,
        "value_data",
        nn.BCELoss(),
        torch.optim.Adam(value_network.parameters(), lr=0.001),
        lambda x: (x > 0.5).long(),
        writer
    )