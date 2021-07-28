from __future__ import print_function
import os,csv
import cv2
import resnet
# import DentNet_ATT
import torch
import numpy as np
import config
import random


projectPath = r"D:\lyc\Project1\November\PaperTest\MetaModel\meta-noftgropnorm\savefile"
SavePath = projectPath + "\\far.txt"
PosPath = projectPath +"\\pos.txt"
NegPath = projectPath + "\\neg.txt"

def get_test_list(pair_list):
    with open(pair_list, 'r',encoding="utf-8") as fd:
        pairs = fd.readlines()
    data_list = []
    label_list = []
    mask_list = []
    for pair in pairs:
        splits = pair.split()
        data_list.append(splits[0])
        mask_list.append(splits[1])
        label_list.append(splits[2])

    return data_list,mask_list,label_list

def dataprocess(img):
    # img = np.array(img) #pil数据转换为np array
    mean = 50
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img[img < mean] = mean
    img = (img - mean) / (255 - mean)
    img = img * 255 #对比度增强
    # img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel) #腐蚀膨胀
    img = img.astype(np.uint8)
    # image = Image.fromarray(img)

    return img

def load_image(img_path,mask_path):
    image = cv2.imread(img_path, 0)
    image = dataprocess(image)
    image = cv2.resize(image,(128,128))
    if image is None:
        return None
    image = np.reshape(image, (128,128,1))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    #处理mask
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (128, 128))
    if mask is None:
        return None
    mask = np.reshape(mask, (128, 128, 1))
    mask = mask.transpose((2,0,1))
    mask = mask[:, np.newaxis, :, :]
    #将mask和原图片合并通道
    image = np.concatenate((image, mask), axis=1)
    image = image.astype(np.float32, copy=False)
    #归一化，可有可无
    image -= 127.5
    image /= 127.5

    return image

def get_featurs(model, test_list, mask_list, batch_size):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path,mask_list[i])
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(device)
            output = model(data)
            output = output.data.cpu().numpy()

            feature = output
            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt

def load_model(model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])

    return model

def get_feature_dict(test_list, features):
    fe_dict = {}
    # f = open(r"./fea_dict.csv", "w", newline="")
    # headers = ["pic", "feature"]
    # f_csv = csv.DictWriter(f, headers)
    # f_csv.writeheader()
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
        # rows = [{"pic":each,"feature":features[i]}]
        # f_csv.writerows(rows)
    return fe_dict

def cosinDistance(Vector1,Vector2):
    """probVectors/gallaryVector is List -> []"""
    v1 = np.array(Vector1)
    v1T = v1.T
    v2 = np.array(Vector2)
    v2T = v2.T
    num = np.dot(v1,v2)
    denum = np.sqrt(np.dot(v1T,v1)*np.dot(v2T,v2))
    cos = num/denum
    return cos

def generate_positive(feature, imgPath, testLabel):
    testLabel = np.array(testLabel)
    num = testLabel.shape[0]
    f = open(PosPath,"w")
    for j in range(num):
        for k in range(j+1, num):
            if testLabel[j] == testLabel[k]:
                f1 = feature[j, :]
                f2 = feature[k, :]
                cosDi = cosinDistance(f1, f2)
                f.write(str(imgPath[j])+" "+str(imgPath[k])+" "+str(cosDi)+"\n")
    f.close()
    print("generate positive score Done!----1/3")

def generate_negtive(feature,imgPath, testLabel):
    testLabel = np.array(testLabel)
    num = testLabel.shape[0]
    f = open(NegPath,"w")
    for j in range(num):
        for k in range(j+1, num):
            if (testLabel[j] == testLabel[k]):
                continue
            else:
                f1 = feature[j, :]
                f2 = feature[k, :]
                cosDi = cosinDistance(f1, f2)
                f.write(str(imgPath[j])+" "+str(imgPath[k])+" "+str(cosDi)+"\n")
    f.close()
    print("generate negtive score Done!----2/3")

def readScore(PosPath, NegPath):
    PostiveScore = []
    NegtiveScore = []
    with open(PosPath, "r") as f:
        temp = [x for x in f]
        for x in temp:
            [label0, label1, score] = x.split()
            PostiveScore.append(float(score))
    f.close()
    with open(NegPath, "r") as t:
        temp0 = [y for y in t]
        for y in temp0:
            [label2, label3, score1] = y.split()
            NegtiveScore.append(float(score1))
    t.close()
    return [PostiveScore, NegtiveScore]

def PostiveRate(threshold, PostiveScore):
    count = 0
    counta = 0
    for x in PostiveScore:
        if x >= threshold:
            counta += 1
        count += 1
    tpr = float(counta) / count
    return tpr

def NegtiveRate(threshold, NegtiveScore):
    count = 0
    counta = 0
    for x in NegtiveScore:
        if x >= threshold:
            counta += 1
        count += 1
    far = float(counta) / count
    return far
def TAR(threshold, PostiveScore):
    count = 0
    counta = 0
    for x in PostiveScore:
        if x <= threshold:
            counta += 1
        count += 1
    frr = float(counta) / count

    return 1-frr

if __name__ == "__main__":

    opt = config.Config()
    device = torch.device("cuda:"+opt.gpu)
    if opt.backbone == 'resnet18':
        model = resnet.resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet.resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet.resnet50()
    # elif opt.backbone == "DentNet_ATT":
    #     model = DentNet_ATT.DentNet_ATT()

    save = open(SavePath, "w")
    model.to(device)
    imgpath_XT, mask_XT, label_XT = get_test_list(opt.Far_list)
    # imgpath,label = get_test_list(opt.FilenameXT)
    model_list = []
    for root,dirs,files in os.walk(opt.test_model_list):
        for file in files:
            model_path = os.path.join(root,file)
            load_model(model, model_path)
            model.eval()
            # features,_ = get_featurs(model,imgpath,opt.test_batch_size)
            features_XT, _ = get_featurs(model, imgpath_XT, mask_XT, opt.test_batch_size)
            # fe_dict = get_feature_dict(imgpath, features)
            generate_positive(features_XT, imgpath_XT, label_XT)
            generate_negtive(features_XT, imgpath_XT, label_XT)

            PostiveScore = []
            NegtiveScore = []
            PostiveScore, NegtiveScore = readScore(PosPath, NegPath)

            thresholdList = list(range(-1000, 1000))  # 产生-1000到999的2000个数
            thresholdx = [float(x) / 1000 for x in thresholdList]  # 改为-1到0.999
            for x in thresholdx:
                Tpr = PostiveRate(x, PostiveScore)
                Far = NegtiveRate(x, NegtiveScore)

                save.write(str(x)+" "+str(Tpr)+" "+str(Far)+"\n")
    save.close()
    print("All Done!")