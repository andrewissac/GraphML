import json
import pickle
import matplotlib
import matplotlib.pyplot as plt
from os import path
from PPrintable import PPrintable
from sklearn.metrics import roc_curve, auc

matplotlib.rcParams.update({'font.size': 16})

class TrainResults(PPrintable):
    def __init__(self):
        self.epochIdx = 0
        self.epoch = []
        self.epochloss = []
        self.runningloss = []

        self.train_result = []
        self.train_acc = []

        self.test_result = []
        self.test_acc = []

        self.val_result = []
        self.val_acc = []

        self.best_val_acc = 0
        self.best_test_acc = 0
        self.best_val_acc_epoch = -1
        self.best_result = {}

        self.summary = []
        self.roc_auc = -1

    def addRunningLoss(self, runningLoss):
        self.runningloss.append(runningLoss)

    def addEpochResult(self, epochloss, train_result, val_result, test_result):
        self.epoch.append(self.epochIdx)
        self.epochloss.append(epochloss)

        self.train_result.append(train_result)
        trainAcc = train_result['acc']
        self.train_acc.append(trainAcc)

        self.val_result.append(val_result)
        valAcc = val_result['acc']
        self.val_acc.append(valAcc)

        self.test_result.append(test_result)
        testAcc = test_result['acc']
        self.test_acc.append(testAcc)

        # Save the best validation accuracy and the corresponding test accuracy.
        if self.best_val_acc < valAcc:
            self.best_val_acc = valAcc
            self.best_val_acc_epoch = self.epochIdx
            self.best_test_acc = testAcc
            self.best_result = {
                'train': self.train_result[self.epochIdx], 
                'val': self.val_result[self.epochIdx], 
                'test': self.test_result[self.epochIdx]
                }
            self.roc_auc = self.getAUC()

        self.summary.append(f'Epoch: {self.epochIdx}, '
            f'Loss: {epochloss:.4f}, '
            f'Train: {trainAcc:.3f}, '
            f'Valid: {valAcc:.3f}, '
            f'Test: {testAcc:.3f}, '
            f'AUC: {self.roc_auc:.3f}')

        self.epochIdx += 1

    def dumpSummary(self, outputPath):   
        with open(path.join(outputPath, 'summary.json'), 'w') as f:
            f.write(json.dumps(self.summary))

    def printLastResult(self):
        self._printResult(-1)
    
    def printBestResult(self):
        print('Best epoch: ')
        self._printResult(self.best_val_acc_epoch)

    def _printResult(self, idx):
        print(f'Epoch: {self.epoch[idx]}, '
            f'Loss: {self.epochloss[idx]:.4f}, '
            f'Train: {self.train_acc[idx]:.3f}, '
            f'Valid: {self.val_acc[idx]:.3f}, '
            f'Test: {self.test_acc[idx]:.3f}, '
            f'AUC: {self.roc_auc:.3f}')

    def saveLossPlot(self, outputPath):
        lineWidth=2
        plt.figure(figsize=(10,7))
        plt.plot(self.epoch, self.epochloss, linewidth=lineWidth)
        plt.title('loss per epoch')
        plt.ylim(0, max(self.epochloss))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        outputFilePath = path.join(outputPath, 'epochloss.jpg')
        plt.savefig(outputFilePath)
        plt.clf()

        plt.figure(figsize=(10,7))
        plt.plot(self.runningloss, linewidth=lineWidth)
        plt.title('running loss')
        plt.ylim(0, max(self.runningloss))
        plt.ylabel('loss')
        plt.xlabel('training step')
        outputFilePath = path.join(outputPath, 'runningloss.jpg')
        plt.savefig(outputFilePath)
        plt.clf()

    def saveAccPlot(self, outputPath):
        lineWidth=2
        plt.figure(figsize=(10,7))
        plt.plot(self.epoch, self.train_acc, label='train acc', linewidth=lineWidth)
        plt.plot(self.epoch, self.val_acc, label='val acc', linewidth=lineWidth)
        plt.plot(self.epoch, self.test_acc, label='test acc', linewidth=lineWidth)
        plt.title('accuracy')
        plt.ylim(0, 1.05)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        outputFilePath = path.join(outputPath, 'accuracy.jpg')
        plt.savefig(outputFilePath)
        plt.clf()

    def saveROCPlot(self, outputPath):
        lineWidth=2
        fpr, tpr = self._getFprTpr()
        roc_auc = auc(fpr, tpr) # AUC = Area Under Curve, ROC = Receiver operating characteristic

        self.roc_auc = roc_auc
        
        plt.figure(figsize=(10,7))
        plt.plot(fpr, tpr, label=f'ROC (area = {roc_auc:.2f})', linewidth=lineWidth)
        plt.plot([0, 1], [0, 1], '--', color='red', label='Luck', linewidth=lineWidth)
        plt.xlabel('False Positive Rate') 
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) curve')
        plt.legend()
        plt.grid()
        outputFilePath = path.join(outputPath, 'ROC.jpg')
        plt.savefig(outputFilePath)
        plt.clf()

    def getAUC(self):
        fpr, tpr= self._getFprTpr()
        roc_auc = auc(fpr, tpr) # AUC = Area Under Curve, ROC = Receiver operating characteristic
        return roc_auc

    def _getFprTpr(self):
        y_true = self.best_result['test']['y_true']
        y_score = self.best_result['test']['y_scoreClass1']

        fpr, tpr, _ = roc_curve(y_true, y_score)
        return fpr, tpr

    def savePlots(self, outputPath):
        self.saveLossPlot(outputPath)
        self.saveAccPlot(outputPath)
        self.saveROCPlot(outputPath)
        plt.close('all')

    def pickledump(self, outputPath):
        with open(path.join(outputPath, 'trainresult.pkl'), 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def loadPickle(filePath):
        with open(filePath, 'rb') as f:
            return pickle.load(f)