from __future__ import print_function
import os
import dataset
import torch
from torch.utils import data
import torch.nn.functional as F
import resnet
import meta
import focal_loss
import metrics
import torchvision
import visualizer
import torch
import numpy as np
import random
import time
import config
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

def save_model(model, save_path, name, iter_cnt):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'loss': loss}
    torch.save(state, save_name)
    # torch.save(model.state_dict(), save_name)
    return save_name

def load_model(model, save_path):
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss = checkpoint['loss']
    else:
        raise FileNotFoundError("Can't find %s"%save_path)

    return model, optimizer, start_epoch, loss
    
def lossfunc(criterion, feature, meta_loss, weight_feature, label, epoch):
    loss_ori = criterion(feature, label)
    loss_mask = meta_loss
    loss_weight = criterion(weight_feature, label)
    if epoch < 60:
        alpha = 0.1 * (1 + epoch // 60)
        beta = 0.6 * (1.3 - epoch // 200)
        loss = loss_ori + alpha * loss_mask + beta * loss_weight
    else:
        loss = loss_ori + 0.1 * loss_mask + 0.8 * loss_weight

    return loss, loss_ori, loss_mask, loss_weight

if __name__ == '__main__':

    opt = config.Config()
    if opt.display:
        visualizer = visualizer.Visualizer(opt.env)

    train_dataset = dataset.Dataset(opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = focal_loss.FocalLoss(gamma=2)
        criterion2 = torch.nn.L1Loss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet.resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet.resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet.resnet50()
    # elif opt.backbone == 'DentNet_ATT':
    #     model = DentNet_ATT.DentNet()
    
    if opt.maskbone == "MetaModel":
        mask_model = meta.Meta()
        

    if opt.metric == 'add_margin':
        metric_fc = metrics.AddMarginProduct(512, opt.num_classes, s=30, m=0.30)
    elif opt.metric == 'arc_margin':
        metric_fc = metrics.ArcMarginProduct(1024, opt.num_classes, opt.gpu, s=30, m=0.7, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = metrics.SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    # print(model)
    # print(mask_model)
    device = torch.device('cuda:'+ opt.gpu if torch.cuda.is_available() else 'cpu')
    model.to(device)
    mask_model.to(device)
    
    metric_fc.to(device)

    if opt.pretrained:
        if opt.continue_train:
            model, optimizer, start_epoch, loss = load_model(model, opt.load_model_path, opt.continue_train)
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            checkpoint = torch.load(opt.load_model_path)
            pretrained_dict = checkpoint['model']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
            for k, v in model.named_parameters():
                if k != 'fc.weight' and k != 'fc.bias'and k != 'bn.weight' and k != 'bn.bias':
                    v.requires_grad = False  # 固定参数
            for k, v in model.named_parameters():
                print(k, v.requires_grad)
            #         print(v.requires_grad)  # 理想状态下，冻结层值都是False
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            start_epoch = 0
    else:
        start_epoch = 0

    if opt.optimizer == "sgd": # filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == "momentum":
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()},{'params': mask_model.parameters()}],
                                    lr=opt.lr, momentum=opt.momentum) # [model.fc5.weight, model.fc5.bias]
    elif opt.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                        lr=opt.lr,alpha=0.9)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)

    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.5)

    start = time.time()
    torch.manual_seed(np.random.randint(1000000))
    for epoch in range(start_epoch,opt.max_epoch): #用于继续训练
        scheduler.step()

        model.train()
        for ii, data in enumerate(trainloader):
            data_input, mask_input, label = data
            data_input = data_input.to(device)
            mask_input = mask_input.to(device)
            label = label.to(device).long()
            label_reshape = label.to(device).float().reshape([label.shape[0],1])
            if opt.display:
                visualizer.img("input",data_input[:1, :, :, :])
                visualizer.img("mask_input", mask_input[:1, :, :, :])
            final_data_input = torch.cat([data_input,mask_input],dim=1)
            feature = model(final_data_input)
            output = metric_fc(feature, label)
            
            mask_feature, mask_logit = mask_model(mask_input)
            meta_loss = criterion2(mask_logit, label_reshape)
            # mask_output = metric_fc(mask_feature, label)
            #权重特征
            weight_feature = feature * mask_feature
            weight_output = metric_fc(weight_feature, label)
            
            loss, loss_ori, loss_mask, loss_weight = lossfunc(criterion, output, meta_loss, weight_output, label, epoch)
            # output = metric_fc(feature, label)
            # loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = epoch * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {:.6f} iters/s lr {:.6f} loss {:.6f} acc {}'.format(time_str, epoch, ii, speed, scheduler.get_lr()[0], loss.item(), acc))
                if opt.display:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.curve(iters, loss_ori.item(), name='loss_original')
                    visualizer.curve(iters, loss_mask.item(), name='loss_mask')
                    visualizer.curve(iters, loss_weight.item(), name='loss_weight')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

        if epoch % opt.save_interval == 0 or epoch == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, epoch)
