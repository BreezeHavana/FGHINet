import os

PosTxt = r"D:\lyc\Project1\July\cross\k_fold\0\savefile\pos.txt"
NegTxt = r"D:\lyc\Project1\July\cross\k_fold\0\savefile\neg.txt"

ab_Pos = r"D:\lyc\Project1\July\cross\k_fold\0\savefile\normal_pos.txt"
ab_Neg = r"D:\lyc\Project1\July\cross\k_fold\0\savefile\ab_neg.txt"

def AnalysePos():
    with open(PosTxt,"r",encoding="utf-8") as f:
        pairs = f.readlines()
    fd = open(ab_Pos,"a",encoding="utf-8")
    for pair in pairs:
        splits = pair.split()
        img1 = splits[0]
        img2 = splits[1]
        label = splits[-1]
        if float(label) >= 0.546:
            fd.writelines(str(img1) + " " +str(img2)+" "+ str(label) + "\n")

    fd.close()
    f.close()

def AnalyseNeg():
    with open(NegTxt,"r",encoding="utf-8") as f:
        pairs = f.readlines()
    fd = open(ab_Neg,"a",encoding="utf-8")
    for pair in pairs:
        splits = pair.split()
        img1 = splits[0]
        img2 = splits[1]
        label = splits[-1]
        if float(label) >= 0.5:
            fd.writelines(str(img1) + " " +str(img2)+" "+ str(label) + "\n")

    fd.close()
    f.close()

if __name__ == '__main__':
    AnalysePos()
    AnalyseNeg()