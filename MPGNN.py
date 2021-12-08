import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import NNConv


class MPGNN(nn.Module):
    """MPGNN.
    MPGNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.
    This class performs message passing in MPGNN and returns the updated node representations.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_in_feats : int
        Size for the input edge features. Default to 128.
    edge_hidden_feats : int
        Size for the hidden edge representations.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    n_classes : int
        Number of graph classes to classify. Default to 1.
    """
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=16,
                 edge_hidden_feats=32, num_step_message_passing=2, n_classes=2):
        super(MPGNN, self).__init__()

        self.project_node_feats = nn.Linear(node_in_feats, node_out_feats)
        self.num_step_message_passing = num_step_message_passing

        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)
        )

        self.gnn_layer = NNConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            edge_func=edge_network,
            aggregator_type='sum'
        )

        self.predict = nn.Linear(node_out_feats, n_classes)
        
    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats.reset_parameters()
        self.gnn_layer.reset_parameters()
        self.predict.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.
        Returns
        -------
        node_feats : float32 tensor of shape (V, node_out_feats)
            Output node representations.
        """
        node_feats = F.relu(self.project_node_feats(node_feats)) # (V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))

        with g.local_scope():
            g.ndata['h'] = node_feats
            # calculate graph representation by average readout
            graph_feats = dgl.mean_nodes(g, 'h')
            return self.predict(graph_feats)


class MPGNN_Mean(nn.Module):
    """MPGNN.
    MPGNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.
    This class performs message passing in MPGNN and returns the updated node representations.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_in_feats : int
        Size for the input edge features. Default to 128.
    edge_hidden_feats : int
        Size for the hidden edge representations.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    n_classes : int
        Number of graph classes to classify. Default to 1.
    """
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=16,
                 edge_hidden_feats=32, num_step_message_passing=2, n_classes=2):
        super(MPGNN_Mean, self).__init__()

        self.project_node_feats = nn.Linear(node_in_feats, node_out_feats)
        self.num_step_message_passing = num_step_message_passing

        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)
        )

        self.gnn_layer = NNConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            edge_func=edge_network,
            aggregator_type='mean'
        )

        self.predict = nn.Linear(node_out_feats, n_classes)
        
    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats.reset_parameters()
        self.gnn_layer.reset_parameters()
        self.predict.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.
        Returns
        -------
        node_feats : float32 tensor of shape (V, node_out_feats)
            Output node representations.
        """
        node_feats = F.relu(self.project_node_feats(node_feats)) # (V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))

        with g.local_scope():
            g.ndata['h'] = node_feats
            # calculate graph representation by average readout
            graph_feats = dgl.mean_nodes(g, 'h')
            return self.predict(graph_feats)