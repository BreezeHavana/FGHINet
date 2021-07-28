import os,cv2,sys,random
import numpy as np
from PIL import Image
import torch,torchvision
from torch.utils import data
from torchvision import transforms as T
import albumentations

from albumentations import (
        Compose,
        OneOf,
        RandomBrightnessContrast,
        MotionBlur,
        MedianBlur,
        GaussianBlur,
        VerticalFlip,
        HorizontalFlip,
        ShiftScaleRotate
)

def dataprocess(img):
    img = np.array(img) #pil数据转换为np array
    mean = 50
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img[img < mean] = mean
    img = (img - mean) / (255 - mean)
    img = img * 255 #对比度增强
    # img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel) #腐蚀膨胀
    img = img.astype(np.uint8)
    image = Image.fromarray(img)

    return image

class Dataset(data.Dataset):
    def __init__(self,data_list_file,phase="train",input_shape=(1,128,128)):
        self.phase = phase
        self.input_shape = input_shape
        self.alb_aug = False

        with open(os.path.join(data_list_file),"r",encoding="utf-8") as fd:
            imgs = fd.readlines()
        self.imgs = [img[:-1] for img in imgs] #图片的绝对路径和标签
        self.imgs = np.random.permutation(imgs)

        normalize = T.Normalize(mean=[0.5],std=[0.5])
        self.train_transforms = Compose(
            [
                # OneOf([RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1,p=1)]),
                OneOf([MotionBlur(blur_limit=3), MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3), ], p=0.5, ),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5)
            ]
        )

        if self.phase == "train":
            self.transforms = T.Compose([
                # T.CenterCrop((650,1700)),
                # T.RandomHorizontalFlip(),
                T.Resize((128,128)),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                # T.CenterCrop((650,1700)),
                T.Resize((128,128)),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        data = Image.open(img_path).convert("L")#(2100,850)
        data = dataprocess(data)
        

        mask_path = splits[1]
        maskB = Image.open(mask_path).convert("L")

        if self.alb_aug:
            augmented = self.train_transforms(image=image_np, mask=maskB_np)
            data = Image.fromarray(augmented['image'])
            maskB = Image.fromarray(augmented['mask'])

        data = self.transforms(data)
        data = data.float()
        maskB = self.transforms(maskB)
        maskB = maskB.float()

        # 随机遮挡图片
        _,height, width = data.shape
        boxh = random.randrange(15, 30) # size500: (30,50)
        boxw = random.randrange(15, 30)
        lh = random.randrange(20, height - 40) # size500: (100,100)
        lw = random.randrange(20, width - 40)
        image_erase = np.ones((height, width,1), np.float)
        image_erase[lh:lh + boxh, lw:lw + boxw,:] = 0
        totensor = T.ToTensor()
        image_erase = totensor(image_erase)
        image_erase = image_erase.float()
        data = data * image_erase

        label = np.int32(splits[2])
        return data,maskB,label
    def __len__(self):
        return len(self.imgs)



if __name__ == '__main__':
    dataset = Dataset(root=r'G:\Tooth_Project\ToothImg\PytorchImg\Train',
                      data_list_file=r'G:\Tooth_Project\Code\Preprocess\new.txt',
                      phase='train',
                      input_shape=(1, 128, 128))

    trainloader = data.DataLoader(dataset,batch_size=1)
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # img = torchvision.utils.save_image(data,"./1.jpg")
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)