import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F

class CustomGCNLayerOnlyNFeatSumMsg(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(CustomGCNLayerOnlyNFeatSumMsg, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature

            # simply copies the node features of all neighbors and puts them into a message
            gcn_msg = fn.copy_u(u='h', out='m') 
            # sums all messages up
            gcn_reduce = fn.sum(msg='m', out='h')

            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

class CustomGCN_OnlyNFeatSumMsg(nn.Module):
    def __init__(self, in_feats, h_feats,num_classes, num_messagepasses=2, dropout=0):
        super(CustomGCN_OnlyNFeatSumMsg, self).__init__()
        self.conv1 = CustomGCNLayerOnlyNFeatSumMsg(in_feats, h_feats)
        self.convLayers = nn.ModuleList()
        for i in range(num_messagepasses - 1):
            self.convLayers.append(CustomGCNLayerOnlyNFeatSumMsg(h_feats, h_feats))
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.predict = nn.Linear(h_feats, num_classes)
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for layer in self.convLayers:
            layer.reset_parameters()
        self.predict.reset_parameters()

    def forward(self, g, in_feat):
        h = F.relu(self.conv1(g, in_feat))
        for layer in self.convLayers:
            if self.dropout != None:
                h = self.dropout(h)
            h = F.relu(layer(g, h))

        with g.local_scope():
            g.ndata['h'] = h
            # calculate graph representation by average readout
            graph_feats = dgl.mean_nodes(g, 'h')
            return self.predict(graph_feats)


class CustomGCNLayerOnlyNFeatMeanMsg(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(CustomGCNLayerOnlyNFeatMeanMsg, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature

            # simply copies the node features of all neighbors and puts them into a message
            gcn_msg = fn.copy_u(u='h', out='m') 
            # mean over all message
            gcn_reduce = fn.mean(msg='m', out='h')

            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

class CustomGCN_OnlyNFeatMeanMsg(nn.Module):
    def __init__(self, in_feats, h_feats,num_classes, num_messagepasses=2, dropout=0):
        super(CustomGCN_OnlyNFeatMeanMsg, self).__init__()
        self.conv1 = CustomGCNLayerOnlyNFeatMeanMsg(in_feats, h_feats)
        self.convLayers = nn.ModuleList()
        for i in range(num_messagepasses - 1):
            self.convLayers.append(CustomGCNLayerOnlyNFeatMeanMsg(h_feats, h_feats))
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.predict = nn.Linear(h_feats, num_classes)
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for layer in self.convLayers:
            layer.reset_parameters()
        self.predict.reset_parameters()

    def forward(self, g, in_feat):
        h = F.relu(self.conv1(g, in_feat))
        for layer in self.convLayers:
            if self.dropout != None:
                h = self.dropout(h)
            h = F.relu(layer(g, h))

        with g.local_scope():
            g.ndata['h'] = h
            # calculate graph representation by average readout
            graph_feats = dgl.mean_nodes(g, 'h')
            return self.predict(graph_feats)