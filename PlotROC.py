# coding=utf-8
import matplotlib.pyplot as plt
import os
from sklearn.metrics import auc

rocPATH1 = r"D:\lyc\Project1\February\Fir\2020-04-17\savefile\far.txt"

SAVENAME = r"D:\lyc\Project1\December\Mask\ROCTooth\savefile\MulROC.png"


def readScore(ROCpath):
    f = open(ROCpath, 'r')
    FPRL = []
    TPRL = []
    tdict = {}
    for str in f:
        __, TPR, FPR = str.split(' ')
        TPRL.append(float(TPR))  # acc
        FPRL.append(float(FPR))  # far
    tdict["TPR"] = TPRL
    tdict["FPR"] = FPRL
    return tdict

def paintImg(temp):
    tpr = temp["TPR"]
    fpr = temp["FPR"]

    flatTpr = []
    Ttemp = tpr[0]
    Ftemp = fpr[0]
    flatFpr = []
    ALPHA = 0.9
    for x in tpr:
        Ttemp = ALPHA * Ttemp + (1 - ALPHA) * x  # 前一个值的0.9倍加后一个值的0.1倍作为新值
        flatTpr.append(Ttemp)
    for x in fpr:
        Ftemp = ALPHA * Ftemp + (1 - ALPHA) * x
        flatFpr.append(Ftemp)
    return flatFpr, flatTpr

if __name__ == "__main__":

    temp = readScore(rocPATH1)  # 字典 acc FAR
    flatFpr, flatTpr = paintImg(temp)
    roc_auc = auc(flatFpr, flatTpr)
    plt.plot(flatFpr, flatTpr, color="red",
            lw=2, label="ours(area = %0.2f)"%roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')#随机曲线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(linestyle='-.')
    plt.legend(loc="lower right")
    plt.savefig("./roc.png",dpi=500,bbox_inches ='tight')
