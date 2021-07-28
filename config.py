class Config(object):
    env = "default"
    gpu = '0'
    backbone = "resnet18" #"DentNet_ATT" #"resnet18"
    maskbone = "MetaModel"
    classify = "softmax"
    num_classes = 9113 #只能和类别数相等
    metric = "arc_margin"
    easy_margin = False
    use_se = False
    loss ="focal_loss"

    display = True
    write = False

    train_list = r"E:\Tooth_Data\data\PytorchImg\Img_mask_train.txt"

    mask_FilenameXT = r"E:\Tooth_Data\data\PytorchImg\Mask_XT.txt"
    mask_FilenameXR = r"E:\Tooth_Data\data\PytorchImg\Mask_XR.txt"
    Far_list = r"E:\Tooth_Data\data\PytorchImg\VerFar.txt"
    
    FirstFile = r"D:\lyc\Project1\November\PaperTest\MetaModel\meta-noftgropnorm"
    
    checkpoints_path = FirstFile + r"\savefile\checkpoints"
    test_model_list = FirstFile + r"\savefile\test"

    pretrained = False
    continue_train = False
    load_model_path = FirstFile + r"\savefile\pretrained\resnet18_25.pth"

    save_interval = 5 #每隔10个epoch保存一次
    train_batch_size = 16 #batch size
    test_batch_size = 4

    input_shape = (1,128,128)
    optimizer = "momentum"
    num_workers = 4 # how many workers for loading data
    print_freq = 100 #print info every N batch

    max_epoch = 85
    lr = 1e-2 #initial learning rate
    lr_step = 30
    lr_decay = 0.5 # when val_loss increase , lr = lr * lr_decay
    weight_decay = 5e-4
    momentum = 0.9

    debug_file = "/tmp/debug" #if os.path.exits(debug_file): enter ipdb
    result_file = "result.csv"




