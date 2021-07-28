from __future__ import print_function
import os, csv
import cv2
import resnet
import torch
import numpy as np
import config

# import DentNet_ATT

FeatureTxt = r"./fea_txt.csv"

counta = 0
count = 0
counta5 = 0
count5 = 0


def get_test_list(pair_list):
    with open(pair_list, 'r', encoding="utf-8") as fd:
        pairs = fd.readlines()
    data_list = []
    label_list = []
    mask_list = []
    for pair in pairs:
        splits = pair.split()
        data_list.append(splits[0])
        mask_list.append(splits[1])
        label_list.append(splits[2])

    return data_list, mask_list, label_list


def dataprocess(img):
    # img = np.array(img) #pil数据转换为np array
    mean = 50
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img[img < mean] = mean
    img = (img - mean) / (255 - mean)
    img = img * 255  # 对比度增强
    # img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel) #腐蚀膨胀
    img = img.astype(np.uint8)
    # image = Image.fromarray(img)

    return img


def load_image(img_path, mask_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        print(img_path)
    image = dataprocess(image)
    image = cv2.resize(image, (128, 128))
    if image is None:
        return None
    image = np.reshape(image, (128, 128, 1))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    # 处理mask
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (128, 128))
    if mask is None:
        return None
    mask = np.reshape(mask, (128, 128, 1))
    mask = mask.transpose((2, 0, 1))
    mask = mask[:, np.newaxis, :, :]
    # 将mask和原图片合并通道
    image = np.concatenate((image, mask), axis=1)
    image = image.astype(np.float32, copy=False)
    # 归一化，可有可无
    image -= 127.5
    image /= 127.5

    return image


def get_featurs(model, test_list, mask_list, batch_size):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path, mask_list[i])
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


def cosinDistance(probVectors, gallaryVector, labelXT, labelXR):
    """
    :param probVectors: XT测试集特征
    :param gallaryVector: XR注册集特征
    :param labelXT: 测试集标签
    :param labelXR: 注册集标签
    :return:
    """
    global count
    global counta
    global count5
    global counta5
    probVectors = np.vstack((probVectors))
    gallaryVector = gallaryVector.T  # (512,5)
    # print(probVectors.shape) # (512,1)
    for j in range(probVectors.shape[1]):
        probstemp = probVectors[:, j]  # (512,1)
        probstempT = probstemp.T  # (1,512)
        lrs = np.zeros((1, gallaryVector.shape[1]))  # 1x
        lrs2 = np.zeros((1, gallaryVector.shape[1]))
        # print(lrs.shape,lrs2.shape)
        for i in range(gallaryVector.shape[1]):
            gallarystemp = gallaryVector[:, i]
            # rows = [{"pic": str(i), "feature": gallarystemp}]
            # f_csv.writerows(rows)
            gallarytempT = gallarystemp.T  # (1,512)
            num = np.dot(probstempT, gallarystemp)  # (1,1)
            denum = np.sqrt(np.dot(probstempT, probstemp) * np.dot(gallarytempT, gallarystemp))
            cos = 1 - num / denum
            lrs[0][i] = cos
            lrs2[0][i] = 1 - cos

        # top1
        minD = np.argmin(lrs, axis=1)
        if labelXR[minD[0]] == labelXT:
            counta += 1
        #     print("top1预测：", labelXR[minD[0]], "真实", labelXT)
        # else:
        #     print("top1预测：", labelXR[minD[0]], "真实", labelXT)
        count += 1
        distance = np.array(lrs2[0])
        minD5 = np.argpartition(distance, -5)[-5:]  # 余弦相似度最大的下标
        for x in range(5):
            if labelXR[minD5[x]] == labelXT:
                counta5 += 1
                break
            """ # 需要时打印
            #else:
                #print("预测top5：", labelXR[minD5[x]], "真实", labelXT[i])
            """
        count5 += 1  # top5比较次数加1

    return lrs


if __name__ == '__main__':

    opt = config.Config()
    device = torch.device("cuda:" + opt.gpu)
    if opt.backbone == 'resnet18':
        model = resnet.resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet.resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet.resnet50()
    # elif opt.backbone == 'DentNet_ATT':
    #     model = DentNet_ATT.DentNet()

    model.to(device)
    imgpath_XR, mask_XR, label_XR = get_test_list(opt.mask_FilenameXR)
    imgpath_XT, mask_XT, label_XT = get_test_list(opt.mask_FilenameXT)
    model_list = []
    for root, dirs, files in os.walk(opt.test_model_list):
        for file in files:
            acc_file = open(r".\\savefile\\" +str(file) + ".txt", "w", encoding="utf-8")
            model_path = os.path.join(root, file)
            load_model(model, model_path)
            model.eval()
            features_XR, _ = get_featurs(model, imgpath_XR, mask_XR, opt.test_batch_size)
            # fe_dict_XR = get_feature_dict(imgpath_XR,features_XR)
            features_XT, _ = get_featurs(model, imgpath_XT, mask_XT, opt.test_batch_size)
            # fe_dict_XT = get_feature_dict(imgpath_XT,features_XT)

            cos_array = np.zeros([len(imgpath_XT), len(imgpath_XR)])
            for i in range(len(imgpath_XT)):
                singlecos = cosinDistance(features_XT[i], features_XR, label_XT[i], label_XR)
                cos_array[i, :] = singlecos

            sort_index = np.argsort(cos_array, axis=1)
            test_cmc =[]
            actual_index = np.array(label_XT, dtype = int)
            predict_index = np.argmin(cos_array, 1)# 返回一列预测标签相对应的最小值的索引值
            temp = np.cast['float32'](np.equal(actual_index, predict_index))#一列相似值，1代表相同，0代表不同
            test_cmc.append(np.mean(temp)) #rank1
            #rank2-rank...
            for i in range(sort_index.shape[1] - 1):
                for j in range(len(temp)):
                    if temp[j] == 0:
                        predict_index[j] = sort_index[j][i + 1]
                temp = np.cast['float32'](np.equal(actual_index, predict_index))
                test_cmc.append(np.mean(temp))
            for ii in test_cmc:
                acc_file.write(str(ii)+"\n")
            # import pylab as plt
            #
            # plt.figure()
            # x = np.arange(0, sort_index.shape[1])
            # plt.plot(x, test_cmc, color="red", linewidth=2)
            # plt.xlabel("Rank")
            # plt.ylabel("Matching Rate")
            # plt.legend()
            # plt.title("CMC Curve")
            # plt.grid(linestyle='-.')
            # plt.savefig("CMC.png", dpi=2000)
            # plt.show()
            acc_file.close()



