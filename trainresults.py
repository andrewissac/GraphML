import json
import pickle
import time
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
        self.epoch_val_loss = []
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

        self.trainingStartTime = 0
        self.trainingEndTime = 0
        self.trainingDuration = 0

        self.fpr = None
        self.tpr = None

    def createFigure(self):
        fig, ax = plt.subplots(figsize=(10,7))
        ax.tick_params(pad=7)
        return fig, ax

    def startTrainingTimer(self):
        self.trainingStartTime = time.time()

    def endTrainingTimer(self):
        self.trainingEndTime = time.time()
        self.trainingDuration = self.trainingEndTime - self.trainingStartTime

    def addRunningLoss(self, runningLoss):
        self.runningloss.append(runningLoss)

    def addEpochResult(self, epochloss, train_result, val_result, test_result):
        self.epoch.append(self.epochIdx)
        self.epochloss.append(epochloss)
        self.epoch_val_loss.append(val_result['loss'])

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
            f'Valid Loss: {val_result["loss"]:.4f}, '
            f'Train: {trainAcc:.3f}, '
            f'Valid: {valAcc:.3f}, '
            f'Test: {testAcc:.3f}, '
            f'AUC: {self.roc_auc:.3f}\n')

        self.epochIdx += 1

    def dumpSummary(self, outputPath):
        self.summary.append("\nBest epoch:\n")
        self.summary.append(self.getBestResult())
        self.summary.append(f"\nThe training took {int(self.trainingDuration)} seconds ({int(self.trainingDuration)/60:.2f} minutes).")

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
        self._printResult(self.best_val_acc_epoch)
        print(f"\nThe training took {int(self.trainingDuration)} seconds ({int(self.trainingDuration)/60:.2f} minutes).")

    def getBestResult(self):
        return self.getResult(self.best_val_acc_epoch)

    def getResult(self, idx):
        return f'Epoch: {self.epoch[idx]}, Loss: {self.epochloss[idx]:.4f}, Validation Loss: {self.epoch_val_loss[idx]:.4f}, Train: {self.train_acc[idx]:.3f}, Validation: {self.val_acc[idx]:.3f}, Test: {self.test_acc[idx]:.3f}, AUC: {self.roc_auc:.3f}'

    def _printResult(self, idx):
        print(self.getResult(idx))

    def saveLossPlot(self, outputPath):
        fig, ax = self.createFigure()
        ax.plot(self.epoch, self.epochloss, label="Training",linewidth=lw)
        ax.plot(self.epoch, self.epoch_val_loss, label="Validation", linewidth=lw)
        ax.set_ylabel('Loss', labelpad=xLabelPad, fontsize=xyLabelFontSize)
        ax.set_xlabel('Epoch', labelpad=yLabelPad, fontsize=xyLabelFontSize)
        ax.legend()
        outputFilePath = path.join(outputPath, 'epochloss.png')
        plt.savefig(outputFilePath)
        plt.clf()

        fig, ax = self.createFigure()
        ax.plot(self.runningloss, label="Training",linewidth=lw)
        ax.set_ylabel('Loss', labelpad=xLabelPad, fontsize=xyLabelFontSize)
        ax.set_xlabel('Gradient step', labelpad=yLabelPad, fontsize=xyLabelFontSize)
        ax.legend()
        outputFilePath = path.join(outputPath, 'runningloss.png')
        plt.savefig(outputFilePath)
        plt.clf()

    def saveAccPlot(self, outputPath):

        fig, ax = self.createFigure()
        ax.plot(self.epoch, self.train_acc, label='Training', linewidth=lw)
        ax.plot(self.epoch, self.val_acc, label='Validation', linewidth=lw)
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

        fig, ax = self.createFigure()
        ax.plot(fpr, tpr, label=f'ROC (area = {roc_auc:.2f})', linewidth=lw)
        ax.plot([0, 1], [0, 1], '--', color='red', label='Luck', linewidth=lw)
        ax.set_xlabel('False Positive Rate') 
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        ax.grid()
        outputFilePath = path.join(outputPath, 'ROC.png')
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
        self.fpr = fpr
        self.tpr = tpr
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