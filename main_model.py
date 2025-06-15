import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader,random_split,ConcatDataset
import torch.optim as optim
import trainer
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
import numpy as np
import random
import os
import wandb
import glob
import argparse
import multiprocessing
from datetime import datetime
from zoneinfo import ZoneInfo
from compute_norm import compute_mean_std
from cad_utils import generate_name,get_handle_front,check_active_block_res50,check_freez_block_res50,UnifiedImageFolderDataset

parser = argparse.ArgumentParser(description='Propert ResNets for smh_dection in pytorch')
parser.add_argument('-p', '--position',nargs = '+', default=None, type=int, metavar='N',
                    help='The position to freeze')
parser.add_argument('-resume', '--resume', action='store_true',
                    help='if exist checkpoint will be use')
parser.add_argument('-epo', '--epoch',default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--src_path', type=str, required=True,help='Source dataset path')
parser.add_argument('--src_name', type=str, default=None,help='Source dataset abbreviation name')

parser.add_argument('-lr',type=float,default=1e-4, metavar='N',
                    help='initial learning rate')
## finetune
parser.add_argument('-ft', '--finetune', action='store_true',
                    help='if exist checkpoint will be use')
parser.add_argument('-ft_lr', '--ft_lr', default=1e-5, type=float, metavar='N',
                    help='the learning rate for finetune')
parser.add_argument('-ft_epo', '--fine_epoch',default=5, type=int, metavar='N',
                    help='number of finetune epochs to run')
parser.add_argument('-ft_p', '--finetune_position',nargs = '+', default=None, type=int, metavar='N',
                    help='The position to freeze')
parser.add_argument('--tar_path', type=str, default=None, help='Target dataset path, only needed for finetuning')
parser.add_argument('--tar_name', type=str, default=None,help='Target dataset abbreviation name')


parser.add_argument('-comb_ds', '--combine_dataset',action='store_true', help='if Combine source and target dataset')

# 记录日志
parser.add_argument("-logger", action='store_true')


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



def setup_paths(args):
    exp_name = generate_name(args)
    now_est = datetime.now(ZoneInfo("America/New_York"))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    full_name = f"{exp_name}_{timestamp}"

    base_dirs = {
        # "log_dir": os.path.join("logs", full_name),
        "ckpt_dir": os.path.join("checkpoints_new", full_name),
        # "result_dir": os.path.join("results", full_name),
    }

    for path in base_dirs.values():
        os.makedirs(path, exist_ok=True)

    return full_name, base_dirs

def get_num_workers():
    try:
        return min(8, multiprocessing.cpu_count())
    except:
        return 0



def main():
    args = parser.parse_args()
    set_seed()
    exp_name, paths = setup_paths(args)

    if args.logger:
        logger = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        # Set the wandb project where this run will be logged.
        project="smh_detection_training",
        # Track hyperparameters and run metadata.
        config=vars(args),
        name = exp_name,
        )   
    else:
        logger = None

    epochs = args.epoch
    # 数据增强与预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
       
    ])

    # 加载数据集
    srcdataset, tardataset = None, None  # 先初始化为 None

    src_path = args.src_path #'/home/shared_data/salmonella_detection/OriginalData/AmericanData'
    tar_path = args.tar_path #'/home/shared_data/salmonella_detection/OriginalData/AmericanData'
    generator = torch.Generator().manual_seed(42)

    if os.path.exists(src_path):
        srcdataset = datasets.ImageFolder(root=src_path, transform=transform)
        # mean, std = compute_mean_std(srcdataset)
        # print("Mean:", mean)                #Mean: tensor([0.5586, 0.5077, 0.4405])
        # print("Std:", std)                  #Std: tensor([0.1756, 0.1774, 0.1781])
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5384, 0.5349, 0.5192],  # ResNet 预训练所用的均值方差
                            std=[0.1387, 0.1396, 0.1512])
                            ])
        srcdataset = datasets.ImageFolder(root=src_path, transform=transform)
        train_size = int(0.8 * len(srcdataset))
        val_size = len(srcdataset) - train_size
        train_dataset, val_dataset = random_split(srcdataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=get_num_workers())
        val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=get_num_workers())
    else:
        raise FileNotFoundError(f"Path does not exist: {src_path}")

    if hasattr(args,"finetune") and args.finetune is True:
        if os.path.exists(tar_path):
            tardataset = datasets.ImageFolder(root=tar_path, transform=transform)
            ft_size = int(0.7 * len(tardataset))
            ft_val_size = len(tardataset) - ft_size
            ft_dataset, ft_val_dataset = random_split(tardataset, [ft_size, ft_val_size], generator=generator)
            ft_loader = DataLoader(ft_dataset, batch_size=32, shuffle=True, num_workers=get_num_workers())
            ft_val_loader = DataLoader(ft_val_dataset, batch_size=32, shuffle=False, num_workers=get_num_workers())

        else:
            raise FileNotFoundError(f"Path does not exist: {tar_path}")
    if hasattr(args, 'combine_dataset') and args.combine_dataset is True:
        if os.path.exists(tar_path):

            
            # 合并源数据集和目标数据集
            # 读取 srcdataset 先统一类别
            srcdataset = datasets.ImageFolder(root=src_path, transform=transform)
            # shared_class_to_idx = srcdataset.class_to_idx

            # # 用相同的 class_to_idx 加载目标集
            tardataset = datasets.ImageFolder(root=tar_path, transform=transform)
            # tardataset.class_to_idx = shared_class_to_idx
            combined_dataset = UnifiedImageFolderDataset(src_dataset=srcdataset,tar_dataset=tardataset,transform=transform)

            # tardataset = datasets.ImageFolder(root=tar_path, transform=transform)
            # combined_dataset = ConcatDataset([srcdataset, tardataset])
            train_size = int(0.8 * len(combined_dataset))
            val_size = len(combined_dataset) - train_size
            combined_train_dataset, combined_val_dataset = random_split(combined_dataset, [train_size, val_size], generator=generator)
            train_loader = DataLoader(combined_train_dataset, batch_size=32, shuffle=True, num_workers=get_num_workers())
            val_loader = DataLoader(combined_val_dataset, batch_size=32, shuffle=False, num_workers=get_num_workers())
        else:
            print("Target dataset is not provided for combining.")
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
    if hasattr(args, 'resume') and args.resume is True:
        if  os.path.exists(checkpoint_record): 
            print("checkpoint exist")
            with open(checkpoint_record, "r") as f:
                checkpoint_path = f.read().strip()
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            ft_epochs = args.fine_epoch
            print(f"model loaded from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"checkpoint does not exist")
    # criterion = nn.CrossEntropyLoss()

    if hasattr(args, 'position') and args.position is not None:
        check_freez_block_res50(model,args.position[0]) # freeze the conv layer not bn layer
        check_active_block_res50(model,args.position[0]+1) # unfreeze the rest block
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,eta_min=1e-6)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2,eta_min=1e-6)
    training = trainer.Trainer(model,train_loader,val_loader,device,optimizer,scheduler,logger=logger,save_dir=paths['ckpt_dir'],status='training')
    training.train(epochs)


    
    if hasattr(args,'finetune') and args.finetune is True:
        if hasattr(args,'finetune_position') and args.finetune_position is not None:
            check_freez_block_res50(model,args.finetune_position[0])
            check_active_block_res50(model,args.finetune_position[0]+1)
        optimizer_ft = optim.SGD(model.parameters(),
                            lr=args.ft_lr,                 # 初始学习率
                            momentum=0.9,          # 动量
                            weight_decay=1e-4)     # L2 正则
        ft_epochs = args.fine_epoch

    # Cosine decay 调度器
        scheduler_ft = CosineAnnealingLR(optimizer_ft, T_max=ft_epochs)
        fine_tuning = trainer.Trainer(model, ft_loader, ft_val_loader, device, optimizer_ft, scheduler_ft,  logger,status='fine_tuning',save_dir=paths['ckpt_dir'])
        fine_tuning.train(ft_epochs)

if __name__ == '__main__':
    main()