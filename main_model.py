import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader,random_split
import torch.optim as optim
import trainer
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import random
import os
import wandb
import glob
import argparse
from compute_norm import compute_mean_std
from cad_utils import generate_name,get_handle_front,check_active_block_res50,check_freez_block_res50

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('-p', '--position',nargs = '+', default=None, type=int, metavar='N',
                    help='The position to freeze')
# parser.add_argument('-cmp', '--compression', action='store_true',
#                     help='if compress the activation')
parser.add_argument('-resume', '--resume', action='store_true',
                    help='if exist checkpoint will be use')
# parser.add_argument('-drp', '--drop', nargs = '+', default=None, type=int, metavar='N',
#                     help='if drop the previous layer')
# parser.add_argument('-tol', '--tolerance', nargs = '+', default=1e-3, type=float, metavar='N',
#                     help='the decompression tolerance')
# parser.add_argument('-fzepo', '--freez_epoch', nargs = '+',type=int, metavar='N',
#                     help='number of total epochs to run')
parser.add_argument('-epo', '--epoch',default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-ft_epo', '--fine_epoch',default=10, type=int, metavar='N',
                    help='number of finetune epochs to run')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 如果使用了 GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_checkpoint(model, optimizer, scheduler, device, save_dir):
    checkpoint_path_file = os.path.join(save_dir, "last_checkpoint.txt")

    if not os.path.exists(checkpoint_path_file):
        print("checkpoint doesn't exist")
        return model, optimizer, scheduler, 0  # start_epoch = 0

    # 读取路径
    with open(checkpoint_path_file, "r") as f:
        checkpoint_path = f.read().strip()

    print(f"load checkpoint：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"]

    return model, optimizer, scheduler, start_epoch


def main():
    args = parser.parse_args()
    set_seed()

    logger = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    # Set the wandb project where this run will be logged.
    project="smh_detection_finetune",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.01,
        "architecture": "Resnet50",
        "dataset": "Smhdata",
        "epochs": 10,
        
    },
    name = generate_name(args),
    )   

    epochs = args.epoch
    # 数据增强与预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
       
    ])

    # 加载数据集
    srcdataset, tardataset = None, None  # 先初始化为 None

    src_path = '/home/shared_data/salmonella_detection/AugmentedData/AfricanDataAug'
    tar_path = '/home/shared_data/salmonella_detection/OriginalData/AmericanData'
    generator = torch.Generator().manual_seed(42)

    if os.path.exists(src_path):
        srcdataset = datasets.ImageFolder(root=src_path, transform=transform)
        # mean, std = compute_mean_std(srcdataset)
        # print("Mean:", mean)                #Mean: tensor([0.5586, 0.5077, 0.4405])
        # print("Std:", std)                  #Std: tensor([0.1756, 0.1774, 0.1781])
        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5384, 0.5349, 0.5192],  # ResNet 预训练所用的均值方差
                            std=[0.1387, 0.1396, 0.1512])
                            ])
        srcdataset = datasets.ImageFolder(root=src_path, transform=transform)
        train_size = int(0.9 * len(srcdataset))
        val_size = len(srcdataset) - train_size
        train_dataset, val_dataset = random_split(srcdataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    else:
        raise FileNotFoundError(f"Path does not exist: {src_path}")

    if os.path.exists(tar_path):
        tardataset = datasets.ImageFolder(root=tar_path, transform=transform)
        ft_size = int(0.7 * len(tardataset))
        ft_val_size = len(tardataset) - ft_size
        ft_dataset, ft_val_dataset = random_split(tardataset, [ft_size, ft_val_size], generator=generator)
        ft_loader = DataLoader(ft_dataset, batch_size=32, shuffle=True, num_workers=0)
        ft_val_loader = DataLoader(ft_val_dataset, batch_size=32, shuffle=False, num_workers=0)

    else:
        raise FileNotFoundError(f"Path does not exist: {tar_path}")
    print("类别映射:", srcdataset.class_to_idx)



# 加载预训练的 ResNet50
    model = models.resnet50(pretrained=True)

    # 替换最后一层为二分类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 二分类

    # 如果你有 GPU 可用：
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    save_dir = "checkpoints"
    checkpoint_record = os.path.join(save_dir, "last_checkpoint.txt")

    if not os.path.exists(checkpoint_record) or (hasattr(args, 'position') and args.position is not None):
        print("checkpoint does not exist")
        optimizer = optim.SGD(model.parameters(),
                            lr=0.1,                 # 初始学习率
                            momentum=0.9,          # 动量
                            weight_decay=5e-4)     # L2 正则

        # Cosine decay 调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        training = trainer.Trainer(model, train_loader, val_loader, device, optimizer, scheduler,logger=logger)
        training.train(epochs)

 
    print("checkpoint exist")
    with open(checkpoint_record, "r") as f:
        checkpoint_path = f.read().strip()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    ft_epochs = args.fine_epoch
    print(f"model loaded from {checkpoint_path}")
    # criterion = nn.CrossEntropyLoss()

    if hasattr(args, 'position') and args.position is not None:
        check_freez_block_res50(model,args.position[0]) # freeze the conv layer not bn layer
        check_active_block_res50(model,args.position[0]+1) # unfreeze the rest block
    optimizer_ft = optim.SGD(model.parameters(),
                        lr=1e-4,                 # 初始学习率
                        momentum=0.9,          # 动量
                        weight_decay=5e-4)     # L2 正则

    # Cosine decay 调度器
    scheduler_ft = CosineAnnealingLR(optimizer_ft, T_max=ft_epochs)
    

    fine_tuning = trainer.Trainer(model, ft_loader, ft_val_loader, device, optimizer_ft, scheduler_ft,"ft_checkpoints",logger,status='fine tuning')
    fine_tuning.train(ft_epochs)

if __name__ == '__main__':
    main()