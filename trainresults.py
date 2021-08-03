import json
import pickle
import matplotlib
import matplotlib.pyplot as plt
from os import path
from PPrintable import PPrintable
from sklearn.metrics import roc_curve, auc

matplotlib.rcParams.update({'font.size': 16})
lw = 2
xyLabelFontSize = 20
xLabelPad = 10
yLabelPad = 15

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

        self.best_test_acc = 0
        self.best_test_acc_epoch = -1
        self.best_result = {}

        self.summary = []
        self.roc_auc = -1

    def addRunningLoss(self, runningLoss):
        self.runningloss.append(runningLoss)

    def addEpochResult(self, epochloss, train_result, test_result):
        self.epoch.append(self.epochIdx)
        self.epochloss.append(epochloss)

        self.train_result.append(train_result)
        trainAcc = train_result['acc']
        self.train_acc.append(trainAcc)

        self.test_result.append(test_result)
        testAcc = test_result['acc']
        self.test_acc.append(testAcc)

        # Save the best test accuracy.
        if self.best_test_acc < testAcc:
            self.best_test_acc = testAcc
            self.best_test_acc_epoch = self.epochIdx
            self.best_result = {
                'train': self.train_result[self.epochIdx], 
                'test': self.test_result[self.epochIdx]
                }
            self.roc_auc = self.getAUC()

        self.summary.append(f'Epoch: {self.epochIdx}, '
            f'Loss: {epochloss:.4f}, '
            f'Train: {trainAcc:.3f}, '
            f'Test: {testAcc:.3f}, '
            f'AUC: {self.roc_auc:.3f}\n')

        self.epochIdx += 1

    def dumpSummary(self, outputPath, trainingTimeInSeconds):
        self.summary.append("\nBest epoch:\n")
        self.summary.append(self.getBestResult())
        self.summary.append(f"\nThe training took {int(trainingTimeInSeconds)} seconds ({int(trainingTimeInSeconds)/60:.2f} minutes).")

        with open(path.join(outputPath, 'summary.txt'), 'w') as f:
            f.writelines(self.summary)

        def formatLoss(loss, idx):
            return f'{loss:.4f}\n'
            
        with open(path.join(outputPath, 'runningloss.txt'), 'w') as f:
            f.writelines([formatLoss(x, ind) for ind, x in enumerate(self.runningloss)])


    def printLastResult(self):
        self._printResult(-1)
    
    def printBestResult(self):
        print('Best epoch: ')
        self._printResult(self.best_test_acc_epoch)

    def getBestResult(self):
        return self.getResult(self.best_test_acc_epoch)

    def getResult(self, idx):
        return f'Epoch: {self.epoch[idx]}, Loss: {self.epochloss[idx]:.4f}, Train: {self.train_acc[idx]:.3f}, Test: {self.test_acc[idx]:.3f}, AUC: {self.roc_auc:.3f}'

    def _printResult(self, idx):
        print(self.getResult(idx))

    def saveLossPlot(self, outputPath):
        fig, ax = plt.subplots(figsize=(9,7))
        ax.tick_params(pad=7)
        ax.plot(self.epoch, self.epochloss, label="Training",linewidth=lw)
        ax.set_ylabel('Loss', labelpad=xLabelPad, fontsize=xyLabelFontSize)
        ax.set_xlabel('Epoch', labelpad=yLabelPad, fontsize=xyLabelFontSize)
        ax.legend()
        outputFilePath = path.join(outputPath, 'epochloss.png')
        plt.savefig(outputFilePath)
        plt.clf()

        fig, ax = plt.subplots(figsize=(9,7))
        ax.tick_params(pad=7)
        ax.plot(self.runningloss, label="Training",linewidth=lw)
        ax.set_ylabel('Loss', labelpad=xLabelPad, fontsize=xyLabelFontSize)
        ax.set_xlabel('Gradient Step', labelpad=yLabelPad, fontsize=xyLabelFontSize)
        ax.legend()
        outputFilePath = path.join(outputPath, 'runningloss.png')
        plt.savefig(outputFilePath)
        plt.clf()

    def saveAccPlot(self, outputPath):

        fig, ax = plt.subplots(figsize=(9,7))
        ax.tick_params(pad=7)
        ax.plot(self.epoch, self.train_acc, label='Training', linewidth=lw)
        ax.plot(self.epoch, self.test_acc, label='Test', linewidth=lw)
        ax.set_ylabel('Accuracy', labelpad=xLabelPad, fontsize=xyLabelFontSize)
        ax.set_xlabel('Epoch', labelpad=yLabelPad, fontsize=xyLabelFontSize)
        ax.legend()
        outputFilePath = path.join(outputPath, 'accuracy.png')
        plt.savefig(outputFilePath)
        plt.clf()

    def saveROCPlot(self, outputPath):
        fpr, tpr = self._getFprTpr()
        roc_auc = auc(fpr, tpr) # AUC = Area Under Curve, ROC = Receiver operating characteristic

        self.roc_auc = roc_auc
        
        plt.figure(figsize=(9,7))
        plt.plot(fpr, tpr, label=f'ROC (area = {roc_auc:.2f})', linewidth=lw)
        plt.plot([0, 1], [0, 1], '--', color='red', label='Luck', linewidth=lw)
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