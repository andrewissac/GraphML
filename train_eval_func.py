import torch
import torch.nn as nn
from TauGraphDataset import GetNodeFeatureVectors, GetEdgeFeatureVectors, GetNeighborNodes, GetEdgeList

# The evaluation function
@torch.no_grad()
def evaluate(model, device, dataloader):
    model = model.to(device)
    model.eval()
    y_true = []
    y_logits = []

    for batched_graph, labels in dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        nodeFeatVec = GetNodeFeatureVectors(batched_graph)

        with torch.no_grad():
            pred = model(batched_graph, nodeFeatVec)

        y_true.append(labels.detach().cpu())
        y_logits.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_logits = torch.cat(y_logits, dim = 0)
    y_softmax = nn.functional.softmax(y_logits, dim=1)
    y_scoreClass1 = y_softmax[:, 1]
    y_pred = y_logits.numpy().argmax(1)
    
    num_correct_pred = (y_pred == y_true).sum().item()
    num_total_pred = len(y_true)
    acc =  num_correct_pred / num_total_pred
    
    evalDict = {
        "y_true": y_true.tolist(), 
        "y_logits": y_logits.tolist(), 
        "y_scoreClass1": y_scoreClass1.tolist(),
        "y_pred": y_pred.tolist(), 
        "acc": acc
    }

    return evalDict


def train(model, device, dataloader, optimizer, loss_fn, batchsize, results):
    model = model.to(device)
    model.train()
    
    epochLoss = 0.0
    batchIter = 0
    
    for batched_graph, labels in dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        nodeFeatVec = GetNodeFeatureVectors(batched_graph)

        #forward
        pred =  model(batched_graph, nodeFeatVec)

        # compute loss
        loss = loss_fn(pred, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # multiply running loss by the number of graphs, 
        # since CrossEntropy loss calculates mean of the losses of the graphs in the batch
        runningTotalLoss = loss.item() #* batchsize
        results.addRunningLoss(runningTotalLoss)
        epochLoss += runningTotalLoss
        batchIter += 1
        
    return epochLoss/batchIter