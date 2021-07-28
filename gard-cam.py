from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2, os
import resnet
import torch.nn as nn
import matplotlib.cm as cm

def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img

def img_preprocess(img_in, mask_or):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    # 对比度增强
    if not mask_or:
        mean = 50
        img[img < mean] = mean
        img = (img - mean) / (255 - mean)
        img = img * 255  # 对比度增强
        img = img.astype(np.uint8)

    else:
        img = img
    img = cv2.resize(img, (128, 128))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],std=[0.5])
    ])

    img_input = img_transform(img, transform)
    return img_input

def draw_CAM(model,img_path,path_mask,save_path,resize=227,isSave=False,isShow=False):
    # 图像加载&预处理
        
    img = cv2.imread(img_path, 0)  # H*W*C
    img_input = img_preprocess(img, False)

    mask = cv2.imread(path_mask, 0)  # H*W*C
    mask_input = img_preprocess(mask, True)
    img_input_final = torch.cat([img_input,mask_input],dim=1)
    
    # 获取模型输出的feature/score
    model.eval() # 测试模式，不启用BatchNormalization和Dropout
    feature = model(img_input_final)
    metric_fc = nn.Sequential(nn.Linear(512 * 8 * 8, 1024), nn.Linear(1024, 9113))
    output = metric_fc(feature.view(1, -1))
    
    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    # 记录梯度值
    def hook_grad(grad):
        global feature_grad
        feature_grad = grad
    feature.register_hook(hook_grad)
    # 计算梯度
    pred_class.backward()
    
    grads=feature_grad # 获取梯度
    
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1)) # adaptive_avg_pool2d自适应平均池化函数,输出大小都为（1，1）

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0] # shape为[batch,通道,size,size],此处batch为1，所以直接取[0]即取第一个batch的元素，就取到了每个batch内的所有元素
    features = feature[0] # 取【0】原因同上
    
    ########################## 导数（权重）乘以相应元素
    # 512是最后一层feature的通道数
    for i in range(len(features)):
        features[i, ...] *= pooled_grads[i, ...] # features[i, ...]与features[i]效果好像是一样的，都是第i个元素
    ##########################
    
    # 绘制热力图
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0) # axis=0,对各列求均值，返回1*n
 
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # 可视化原始热力图
    # if isShow:
    #     plt.matshow(heatmap)
    #     plt.show()
        
    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    # 将图像保存到硬盘
    # if isSave:
    #     # superimposed_img /= 255
    #     cv2.imwrite(save_path, superimposed_img)
    # 展示图像
    if isShow:
        superimposed_img/=255

        # plt.imshow(superimposed_img)
        # plt.axis('off')
        plt.imshow(superimposed_img, cmap='YlOrRd')
        plt.colorbar(shrink=0.41)
        plt.axis('off')
        plt.savefig(save_path)

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pair_list = r"D:\lyc\lcanet\train.txt"
    path_net = os.path.join(BASE_DIR,  "savefile\\test", "resnet18_20.pth")

    model= resnet.resnet_face18(False)
    checkpoint = torch.load(path_net)
    model.load_state_dict(checkpoint['model'])
        
    # draw_CAM(model,'/Users/liuyanzhe/Study/陶瓷研究相关/陶瓷数据/ceramic_data/训练/1牡丹/mudan15.png','/Users/liuyanzhe/Downloads/热力图1.png',isSave=True,isShow=True)

    with open(pair_list, 'r', encoding="utf-8") as fd:
        pairs = fd.readlines()
    data_list = []
    label_list = []
    mask_list = []
    for pair in pairs:
        splits = pair.split()
        name = splits[0].split("\\")[-1]
        output_dir = os.path.join(BASE_DIR, "Result", name)
        draw_CAM(model, splits[0], splits[1], output_dir, isSave=True, isShow=True)
