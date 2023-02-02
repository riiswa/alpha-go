import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from net import FeatureExtractor, GoNetwork, train

if __name__ == "__main__":
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()
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

    feature_network1 = FeatureExtractor(dataset["X"].shape[1], 128, 6).to(device)
    policy_network = GoNetwork(
        feature_network1,
        81,
        nn.functional.softmax,
        lambda x, x_: x * x_[:, -1, :, :].flatten(start_dim=1)
    ).to(device)

    policy_network.load_state_dict(torch.load("weights/network_policy_data_320_weights.pt"))

    feature_network2 = FeatureExtractor(dataset["X"].shape[1], 128, 6).to(device)
    feature_network2.load_state_dict(policy_network.feature_extractor.state_dict())
    for param in feature_network2.parameters():
        param.requires_grad = False

    value_network = GoNetwork(feature_network2, 1, nn.functional.sigmoid).to(device)

    print("Start value network training...")
    train(
        value_network,
        train_dataset,
        test_dataset,
        "value_data",
        nn.BCELoss(),
        torch.optim.Adam(value_network.parameters(), lr=0.005),
        lambda x: (x > 0.5).long(),
        writer,
        device,
        epochs=200
    )