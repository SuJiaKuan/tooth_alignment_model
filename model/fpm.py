import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from utils.const import TEETH as _TEETH


TEETH = _TEETH + [
    "jaw",
]
TOOTH2NODE = {tooth: node for node, tooth in enumerate(TEETH)}


def create_fpm_graph():
    edge_indices = {"source": [], "target": []}

    adjacent_edge_pairs = [
        ["1.7", "1.6"],
        ["1.6", "1.5"],
        ["1.5", "1.4"],
        ["1.4", "1.3"],
        ["1.3", "1.2"],
        ["1.2", "1.1"],
        ["1.1", "2.1"],
        ["2.1", "2.2"],
        ["2.2", "2.3"],
        ["2.3", "2.4"],
        ["2.4", "2.5"],
        ["2.5", "2.6"],
        ["2.6", "2.7"],
        ["3.7", "3.6"],
        ["3.6", "3.5"],
        ["3.5", "3.4"],
        ["3.4", "3.3"],
        ["3.3", "3.2"],
        ["3.2", "3.1"],
        ["3.1", "4.1"],
        ["4.1", "4.2"],
        ["4.2", "4.3"],
        ["4.3", "4.4"],
        ["4.4", "4.5"],
        ["4.5", "4.6"],
        ["4.6", "4.7"],
    ]

    symmetric_edge_pairs = [
        ["1.1", "2.1"],
        ["1.2", "2.2"],
        ["1.3", "2.3"],
        ["1.4", "2.4"],
        ["1.5", "2.5"],
        ["1.6", "2.6"],
        ["1.7", "2.7"],
        ["3.1", "4.1"],
        ["3.2", "4.2"],
        ["3.3", "4.3"],
        ["3.4", "4.4"],
        ["3.5", "4.5"],
        ["3.6", "4.6"],
        ["3.7", "4.7"],
    ]

    jaw_edge_pairs = [
        ["jaw", "1.7"],
        ["jaw", "1.6"],
        ["jaw", "1.5"],
        ["jaw", "1.4"],
        ["jaw", "1.3"],
        ["jaw", "1.2"],
        ["jaw", "1.1"],
        ["jaw", "2.1"],
        ["jaw", "2.2"],
        ["jaw", "2.3"],
        ["jaw", "2.4"],
        ["jaw", "2.5"],
        ["jaw", "2.6"],
        ["jaw", "2.7"],
        ["jaw", "3.7"],
        ["jaw", "3.6"],
        ["jaw", "3.5"],
        ["jaw", "3.4"],
        ["jaw", "3.3"],
        ["jaw", "3.2"],
        ["jaw", "3.1"],
        ["jaw", "4.1"],
        ["jaw", "4.2"],
        ["jaw", "4.3"],
        ["jaw", "4.4"],
        ["jaw", "4.5"],
        ["jaw", "4.6"],
        ["jaw", "4.7"],
    ]

    all_edge_pairs = \
        adjacent_edge_pairs \
        + symmetric_edge_pairs \
        + jaw_edge_pairs

    # Add edges.
    for source, target in all_edge_pairs:
        # Undirected graph
        edge_indices["source"].append(TOOTH2NODE[source])
        edge_indices["target"].append(TOOTH2NODE[target])
        edge_indices["source"].append(TOOTH2NODE[target])
        edge_indices["target"].append(TOOTH2NODE[source])

    return edge_indices


class FeaturePropogationModule(torch.nn.Module):

    def __init__(self, in_features=256, out_features=256, conv="gat"):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.conv = conv

        if self.conv == "gat":
            self.conv_class = GATConv
        elif self.conv == "gcn":
            self.conv_class = GCNConv
        else:
            raise ValueError(
                "Non-Supported graph convolution: {}".format(self.conv),
            )

        self.conv1 = self.conv_class(256, 256)
        self.conv2 = self.conv_class(256, 256)

        self.edge_index = self._create_edge_index()

    def _create_edge_index(self):
        fpm_graph = create_fpm_graph()
        edge_index = torch.tensor(
            [fpm_graph["source"], fpm_graph["target"]],
            dtype=torch.long,
        )

        return edge_index

    def forward(self, fea):
        edge_index = self.edge_index.to(fea.device)

        x = torch.stack([self.conv1(fea_, edge_index) for fea_ in fea], dim=0)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.stack([self.conv2(x_, edge_index) for x_ in x], dim=0)

        return x
