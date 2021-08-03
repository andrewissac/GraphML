import os
import dgl
import math
import glob
import torch
import uproot
import random
import numpy as np
import awkward as ak
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from os import path
from tqdm import tqdm
from functools import reduce
from pathlib import Path
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from TauGraphDatasetInfo import TauGraphDatasetInfo

class TauGraphDataset(DGLDataset):
    """ Template for customizing graph datasets in DGL.
    Parameters
    ----------
    name : str
        Name of the dataset
    rootfiles: dict[int:list[str]]
        Stores a dict with keys as graph labels and 
        values as the list of rootfile paths
    featureMapping: FeatureMapping
        Stores the order of the node/edge/graph features in the feature vectors
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self, 
                 name: str, 
                 save_dir: str,
                 info: TauGraphDatasetInfo=None,
                 shuffleDataset: bool=True, 
                 url=None, 
                 raw_dir=None, 
                 force_reload=False, 
                 verbose=True):
        self.info = info
        self.shuffleDataset = shuffleDataset
        super(TauGraphDataset, self).__init__(name=name,
                                         url=url, 
                                         raw_dir=raw_dir, 
                                         save_dir=save_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)

    def process(self):
        # process raw data to graphs, labels, splitting masks
        self.graphs = []
        self.labels = []
        phi = self.info.nodeFeatMap['phi']
        eta = self.info.nodeFeatMap['eta']
        pt = self.info.nodeFeatMap['pt']

        def shift_to_minuspi_pi(x):
            """
            Shift number into the interval [-pi, pi)
            """
            twoPi = 2*np.pi
            while(x >= np.pi):
                x -= twoPi
            while(x < -np.pi):
                x += twoPi
            return x      

        # generate graphs from rootfiles
        it = 1
        for label, rootfiles in self.info.rootfiles.items():
            graphCount = 0
            for rootfile in rootfiles:
                ttree = uproot.open(rootfile + ':taus')
                pfCand_pt = ttree["pfCand_pt"].array()
                pfCand_eta = ttree["pfCand_eta"].array()
                pfCand_phi = ttree["pfCand_phi"].array()
                pfCand_mass = ttree["pfCand_mass"].array()
                pfCand_charge = ttree["pfCand_charge"].array()
                pfCand_particleType = ttree["pfCand_particleType"].array()
                nodeCounts = [len(x) for x in pfCand_pt]
                pfCand_summand = []
                for nodeCount in nodeCounts:
                    pfCand_summand.append([1 for i in range(nodeCount)])

                for i in tqdm(range(len(nodeCounts)), 
                desc=f'({it}/{self.info.rootfilesCount}) Generating graphs from {Path(rootfile).name}'):
                    # only build as many graphs for this graphClass as specified in the info
                    if graphCount >= self.info.graphsPerClass:
                        break

                    # preselect only events with at least 2 pfCands (nodes)
                    nodeCount = int(nodeCounts[i])
                    if nodeCount < 2:
                        continue

                    # nodeFeature[nFeatMap['Phi']] -> contains a list with ALL Phis from this graph
                    nodeFeatures = []
                    nodeFeatures.append(np.array(pfCand_pt[i]))
                    nodeFeatures.append(np.array(pfCand_eta[i]))
                    nodeFeatures.append(np.array(pfCand_phi[i]))
                    nodeFeatures.append(np.array(pfCand_mass[i]))
                    nodeFeatures.append(np.array(pfCand_charge[i]))
                    nodeFeatures.append(np.array(pfCand_particleType[i]))
                    nodeFeatures.append(np.array(pfCand_summand[i]))

                    src_ids = []
                    dst_ids = []
                    edgeFeatures = []
                    for j in range(nodeCount):
                        for k in range(nodeCount):
                            if not j == k: # no self-loops
                                # add src/dst node ids
                                src_ids.append(j)
                                dst_ids.append(k)

                                deltaEta = np.float32(nodeFeatures[eta][j] - nodeFeatures[eta][k])
                                deltaPhi = np.float32(shift_to_minuspi_pi(nodeFeatures[phi][j] - nodeFeatures[phi][k]))
                                deltaR = np.float32(math.sqrt(deltaPhi * deltaPhi + deltaEta * deltaEta))
                                # add edge features (care, order should be the same as in eFeatMapping!)
                                edgeFeatures.append([deltaEta, deltaPhi, deltaR])

                    # build graph based on src/dst node ids
                    g = dgl.graph((src_ids, dst_ids))
                    # dstack -> each entry is now a node feature vec containing [P_t, Eta, Phi, Mass, Type] for that node
                    nodeFeatures = np.dstack(nodeFeatures).squeeze()
                    g.ndata['feat'] = torch.from_numpy(nodeFeatures)
                    g.edata['feat'] = torch.tensor(edgeFeatures)
                    # order: 1st NodeCount
                    graphFeatures = { 'feat' : torch.from_numpy(np.array([nodeCount]))}
                    setattr(g, 'gdata', graphFeatures)
                    
                    self.graphs.append(g)
                    self.labels.append(label)
                    graphCount += 1
                it += 1

        # shuffle dataset
        if self.shuffleDataset:
            # hacky, better way to shuffle both lists?
            random.seed(0)
            random.shuffle(self.graphs)
            random.seed(0)
            random.shuffle(self.labels)
            
        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

        # save histograms
        #self.saveEtaPtScatterPlot()
        self.printProperties()

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_dir, f'{self.name}_graphs.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # save other information in python dict
        info_path = os.path.join(self.save_dir, f'{self.name}_properties.pkl')
        save_info(info_path,{
            'num_graph_classes' : self.num_graph_classes,
            'graphClasses' : self.graphClasses,
            'num_graphs' : self.num_graphs,
            'num_all_nodes' : self.num_all_nodes,
            'num_all_edges' : self.num_all_edges,
            'dim_nfeats' : self.dim_nfeats,
            'dim_efeats' : self.dim_efeats,
            'dim_gfeats' : self.dim_gfeats,
            'nodeFeatKeys' : self.nodeFeatKeys,
            'edgeFeatKeys' : self.edgeFeatKeys,
            'graphFeatKeys' : self.graphFeatKeys,
            'maxNodeCount' : self.maxNodeCount,
            'minNodeCount' : self.minNodeCount
            })

    def load(self):
        # load processed data from directory `self.save_dir`
        graph_path = os.path.join(self.save_dir, f'{self.name}_graphs.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']

        datasetinfo_path = os.path.join(self.save_dir, f'{self.name}.json')
        self.info = TauGraphDatasetInfo.LoadFromJsonfile(datasetinfo_path)
        # propertyinfo_path = os.path.join(self.save_dir, f'{self.name}_properties.pkl')
        # loadedInfos = load_info(propertyinfo_path)

    def has_cache(self):
        # check whether there are processed data in `self.save_dir`
        graph_path = os.path.join(self.save_dir, f'{self.name}_graphs.bin')
        propertyinfo_path = os.path.join(self.save_dir, f'{self.name}_properties.pkl')
        datasetinfo_path = os.path.join(self.save_dir, f'{self.name}.json')
        return os.path.exists(graph_path) and os.path.exists(propertyinfo_path) and os.path.exists(datasetinfo_path)
    
    @property
    def num_graph_classes(self):
        return self.info.numberOfGraphClasses

    @property
    def graphClasses(self):
        return list(self.info.rootfiles.keys())
    
    @property
    def num_graphs(self):
        return len(self.graphs)
    
    @property
    def num_all_nodes(self):
        return sum([g.number_of_nodes() for g in self.graphs])
    
    @property
    def num_all_edges(self):
        return sum([g.number_of_edges() for g in self.graphs])

    @property
    def nodeFeatKeys(self):
        return list(self.info.nodeFeatMap.keys())

    @property
    def edgeFeatKeys(self):
        return list(self.info.edgeFeatMap.keys())

    @property
    def graphFeatKeys(self):
        return list(self.info.graphFeatMap.keys())
    
    @property
    def dim_nfeats(self):
        return len(self.nodeFeatKeys)

    @property
    def dim_efeats(self):
        return len(self.edgeFeatKeys)

    @property
    def dim_gfeats(self):
        return len(self.graphFeatKeys)

    @property
    def maxNodeCount(self):
        return max([g.number_of_nodes() for g in self.graphs])

    @property
    def minNodeCount(self):
        return min([g.number_of_nodes() for g in self.graphs])
    
    def get_split_indices(self):
        train_split_idx = int(len(self.graphs) * self.info.splitPercentages['train'])
        return {
            'train': torch.arange(train_split_idx),
            'test': torch.arange(train_split_idx, len(self.graphs))
        }
    
    def download(self):
        # download raw data to local disk
        pass
    
    def printProperties(self):
        print(f'Num Graph classes: {self.num_graph_classes}')
        print(f'Graph classes: {self.graphClasses}')
        print(f'Number of graphs: {self.num_graphs}')
        print(f'Number of all nodes in all graphs: {self.num_all_nodes}')
        print(f'Number of all edges in all graphs: {self.num_all_edges}')
        print(f'Dim node features: {self.dim_nfeats}')
        print(f'Node feature keys: {self.nodeFeatKeys}')
        print(f'Dim edge features: {self.dim_efeats}')
        print(f'Edge feature keys: {self.edgeFeatKeys}')
        print(f'Dim graph features: {self.dim_gfeats}')
        print(f'Graph feature keys: {self.graphFeatKeys}')
        print(f'Max node count: {self.maxNodeCount}')
        print(f'Min node count: {self.minNodeCount}')

    def _getFeatureByKey(self, g, key):
        if(key in self.info.nodeFeatMap):
            return g.ndata['feat'][:,self.info.nodeFeatMap[key]].tolist()
        elif(key in self.info.edgeFeatMap):
            return g.edata['feat'][:,self.info.edgeFeatMap[key]].tolist()
        else:
            raise KeyError(f'Key {key} not found in node or edge data.')

    def _accumulateFeature(self, key, graphLabel):
        """
        Goes through all graphs, concats the specified feature (by key) 
        if the graphLabel matches and returns the tensor
        would be better to use masks instead of iterating 
        """
        if self.num_graphs <= 0:
            raise Exception('There are no graphs in the dataset.')
        accumulatedFeat = []
        for i in range(self.num_graphs):
            if self.labels[i] == graphLabel:
                feat = self._getFeatureByKey(self.graphs[i], key)
                accumulatedFeat.extend(feat)
        return accumulatedFeat

    def saveEtaPtScatterPlot(self, outputPath=''):
        data = { 'pt' : [], 'eta' : []}

        # iterate through all graphClasses
        for gclass in self.graphClasses:
            pt = self._accumulateFeature('pt', gclass)
            eta = self._accumulateFeature('eta', gclass)
            data['pt'].append((gclass, pt))
            data['eta'].append((gclass, eta))

        fig, ax = plt.subplots(figsize=(10,7))
        ax.scatter(data['eta'][0][1], data['pt'][0][1], label=f'GraphClass0', alpha=0.3)
        ax.scatter(data['eta'][1][1], data['pt'][1][1], label=f'GraphClass1', alpha=0.3)
        ax.set_xlabel('eta')
        ax.set_ylabel('pt')
        ax.legend()
        if outputPath == '':
            outputPath = self.save_dir
        outputFilePath = path.join(outputPath, self.name + '_Pt_Eta_ScatterPlot.jpg')
        fig.savefig(outputFilePath)
        plt.clf()
        plt.cla()
        plt.close()

def GetNodeFeatureVectors(graph):
    #print(graph.ndata.values())
    feat = []
    for key, val in graph.ndata.items():
        if key != 'label':
            feat.append(val)
    #print(feat)
    return _getFeatureVec(feat)

def GetEdgeFeatureVectors(graph):
    return graph.edata['feat']

def GetEdgeFeatureVectorsFromSourceNode(graph, sourceNodeID):
    allEdgeFeat = GetEdgeFeatureVectors(graph)
    nodeCount = graph.num_nodes()
    beginIdx = sourceNodeID * (nodeCount-1)
    endIdx = beginIdx + nodeCount-1
    return allEdgeFeat[beginIdx:endIdx]

def _getFeatureVec(data):
    feat = tuple([x.data for x in data])
    feat = torch.dstack(feat).squeeze()
    feat = feat.float()
    return feat

def GetNeighborNodes(graph, sourceNodeLabel: int):
    """
    returns a tuple(tensor(srcNodeIDs), tensor(dstNodeIDs)) of the whole graph
    """
    return graph.out_edges(sourceNodeLabel)

def GetEdgeList(graph):
    """
    returns a tuple(tensor(srcNodeID), tensor(dstNodeId)) of the whole graph
    """
    return graph.edges(form='uv', order='srcdst')

def Graph2FlatZeropaddedList(graph, nFeatDim, eFeatDim, maxNodeCount, useEdgeFeat):
    """
    Flatten + zero pad graph data to be able to use it as an input feature vector in keras.
    
    The zero padding always depends on the graph with the highest node count in the dataset
    example: biggest graph in a graph dataset might has maxNodeCount=81 nodes
    The graph data that needs to be converted into a 1D array (which contains node+edge features) has 25 nodes  
    and needs to be padded with zeros until 81 nodes. Always keep in mind that we are using fully connected graphs.
    
    Lets say we have:
    nodefeat_dim: 7
    edgefeat_dim: 3
    nodeCount: 25
    maxNodeCount = 81
    
    That means we will have to zero pad for 3*81*80=19440 edge features and 7*81=567 node features!
    ------
    The padded 1D array information about a single node has a length of:
    (nodefeat_dim + edgefeat_dim * (nodeCount-1) + edgefeat_dim * ((maxNodeCount-1) - (nodeCount-1)))
    = 7 + 3 * (25-1) + 3 * ((81-1) - (25-1)) = 7 + 3 * 24 + 3 * 56 = 7 + 3 * 80 = 247
    
    The padded 1D array information for the whole graph will ultimately have the length of:
    81 * 247 = 20007 = 19440 (edge features) + 567 (nodefeatures)
    """
    kerasInput = []
    nodeCount = graph.num_nodes()
    nFeats = graph.ndata['feat']
    eFeats = graph.edata['feat']

    def _getEdgeFeatureVectorsFromSourceNode(allEdgeFeats, nodeCount, sourceNodeID):
        beginIdx = sourceNodeID * (nodeCount-1)
        endIdx = beginIdx + nodeCount-1
        return allEdgeFeats[beginIdx:endIdx]

    for i in range(maxNodeCount):
        if i < nodeCount:
            kerasInput.extend(torch.flatten(nFeats[i]))
            if useEdgeFeat:
                sourceNodeEdgeFeatures = torch.flatten(_getEdgeFeatureVectorsFromSourceNode(eFeats, nodeCount, i))
                kerasInput.extend(sourceNodeEdgeFeatures)
                zerosNeeded = (maxNodeCount-1) - (nodeCount-1)
                kerasInput.extend([0.0] * eFeatDim * zerosNeeded)
        elif nodeCount <= i < maxNodeCount: # fill everything with zeros until maxNodeCount
            kerasInput.extend([0.0] * nFeatDim)
            if(useEdgeFeat):
                kerasInput.extend([0.0] * eFeatDim * (maxNodeCount - 1))
    return kerasInput