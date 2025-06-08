import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torchvision import transforms, models, datasets
from tqdm import tqdm
from datetime import datetime


# ========== 数据加载器 ==========
def build_dataloaders(src_path, tar_path, batch_size, generator, transform=None):
    # 数据预处理
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5384, 0.5349, 0.5192], std=[0.1387, 0.1396, 0.1512])  # ResNet 预训练的均值和方差
        ])
    
    # 处理源数据集（src_path）——用于训练
    if os.path.exists(src_path):
        srcdataset = datasets.ImageFolder(root=src_path, transform=transform)
        # 分割训练集和验证集
        train_size = int(0.9 * len(srcdataset))
        val_size = len(srcdataset) - train_size
        train_dataset, val_dataset = random_split(srcdataset, [train_size, val_size], generator=generator)

        # 创建训练和验证 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        raise FileNotFoundError(f"Path does not exist: {src_path}")
    
    # 处理目标数据集（tar_path）——用于微调
    if os.path.exists(tar_path):
        tardataset = datasets.ImageFolder(root=tar_path, transform=transform)
        ft_size = int(0.7 * len(tardataset))  # 70% 用于微调，30% 用于验证
        ft_val_size = len(tardataset) - ft_size
        ft_dataset, ft_val_dataset = random_split(tardataset, [ft_size, ft_val_size], generator=generator)

        # 创建微调训练和验证 DataLoader
        ft_loader = DataLoader(ft_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        ft_val_loader = DataLoader(ft_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        raise FileNotFoundError(f"Path does not exist: {tar_path}")

    # 输出类别映射
    print("类别映射:", srcdataset.class_to_idx)

    return train_loader, val_loader, ft_loader, ft_val_loader

# ========== 模型构建 ==========
def build_model(num_classes=2):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ========== Trainer 类 ==========

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, optimizer, scheduler, save_dir="checkpoints", logger=None,status='training'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.save_dir = save_dir
        self.best_val_acc = 0.0
        self.logger = logger
        self.status = status

        os.makedirs(self.save_dir, exist_ok=True)

    def train_one_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    def save_model(self, epoch, is_best=False):
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 决定文件名：最佳模型 or 普通 epoch 模型
        filename = f"best_model_{time_str}.pth" if is_best else f"epoch_{epoch+1}_{time_str}.pth"
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        with open(os.path.join(self.save_dir, "last_checkpoint.txt"), "w") as f:
            f.write(path)

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            print(self.status,f"[Epoch {epoch+1}] "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            if self.logger is not None:
                self.logger.log({"Train Loss":train_loss,"Train Acc":train_acc,"Val loss":val_loss,"Val acc":val_acc})

            # 保存当前 epoch 的模型
            if (epoch + 1) % 10 == 0:
                self.save_model(epoch)

            # 如果当前验证准确率是最佳，保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(epoch, is_best=True)
                test=0
