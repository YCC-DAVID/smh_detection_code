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





def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 如果使用了 GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
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
    )   

    epochs = 1
    # 数据增强与预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5586, 0.5077, 0.4405],  # ResNet 预训练所用的均值方差
                            std=[0.1756, 0.1774, 0.1781])
    ])

    # 加载数据集
    srcdataset, tardataset = None, None  # 先初始化为 None

    src_path = 'AfricanData'
    tar_path = 'AmericanData'
    generator = torch.Generator().manual_seed(42)

    if os.path.exists(src_path):
        srcdataset = datasets.ImageFolder(root=src_path, transform=transform)
        train_size = int(0.9 * len(srcdataset))
        val_size = len(srcdataset) - train_size
        train_dataset, val_dataset = random_split(srcdataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    else:
        raise FileNotFoundError(f"Path does not exist: {src_path}")

    if os.path.exists(tar_path):
        tardataset = datasets.ImageFolder(root=tar_path, transform=transform)
        ft_size = int(0.9 * len(tardataset))
        ft_val_size = len(tardataset) - train_size
        ft_dataset, ft_val_dataset = random_split(tardataset, [ft_size, ft_val_size], generator=generator)
        ft_loader = DataLoader(ft_dataset, batch_size=32, shuffle=True, num_workers=2)
        ft_val_loader   = DataLoader(ft_val_dataset, batch_size=32, shuffle=False, num_workers=2)

    else:
        raise FileNotFoundError(f"Path does not exist: {tar_path}")
        # ================================
        # 4. 按 9:1 比例划分 train/val
        # ================================
        

# 使用固定种子进行划分以确保可复现
    

    
    
    # 创建 DataLoader
    

    


    print("类别映射:", srcdataset.class_to_idx)



# 加载预训练的 ResNet50
    model = models.resnet50(pretrained=True)

    # 替换最后一层为二分类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 二分类

    # 如果你有 GPU 可用：
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    checkpoint_path = "checkpoint.pth"
    if not os.path.exists(checkpoint_path):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                            lr=0.1,                 # 初始学习率
                            momentum=0.9,          # 动量
                            weight_decay=5e-4)     # L2 正则

        # Cosine decay 调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        training = trainer.Trainer(model, train_loader, val_loader, device, optimizer, scheduler)
        training.train(epochs)
    else:
        epochs = 1
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"model loaded from {checkpoint_path}")
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(model.parameters(),
                            lr=0.01,                 # 初始学习率
                            momentum=0.9,          # 动量
                            weight_decay=5e-4)     # L2 正则

        # Cosine decay 调度器
        scheduler = CosineAnnealingLR(optimizer_ft, T_max=epochs)

        fine_tuning = trainer.Trainer(model, ft_loader, ft_val_loader, device, optimizer_ft, scheduler,"ft_checkpoints")
        fine_tuning.train(epochs)
    