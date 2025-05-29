import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader,random_split
import torch.optim as optim
import trainer
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import random

a=0


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

    epochs = 100

    # 数据增强与预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5586, 0.5077, 0.4405],  # ResNet 预训练所用的均值方差
                            std=[0.1756, 0.1774, 0.1781])
    ])

    # 加载数据集
    dataset = datasets.ImageFolder(root='AmericanData', transform=transform)

    # ================================
    # 4. 按 9:1 比例划分 train/val
    # ================================
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

# 使用固定种子进行划分以确保可复现
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print("类别映射:", dataset.class_to_idx)



# 加载预训练的 ResNet50
    model = models.resnet50(pretrained=True)

    # 替换最后一层为二分类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 二分类

    # 如果你有 GPU 可用：
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                        lr=0.1,                 # 初始学习率
                        momentum=0.9,          # 动量
                        weight_decay=5e-4)     # L2 正则

    # Cosine decay 调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    training = trainer.Trainer(model, train_loader, val_loader, device, optimizer, scheduler)
    training.train(epochs)
    