import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torchvision import transforms, models, datasets
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from typing import List, Tuple
from datetime import datetime


# ========== 数据加载器 ==========
def build_dataloaders(src_path, tar_path, batch_size, generator):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5384, 0.5349, 0.5192],
                             std=[0.1387, 0.1396, 0.1512])
    ])

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source path does not exist: {src_path}")

    srcdataset = datasets.ImageFolder(root=src_path, transform=transform)
    train_size = int(0.8 * len(srcdataset))
    val_size = len(srcdataset) - train_size
    train_dataset, val_dataset = random_split(srcdataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    ft_loader = ft_val_loader = None

    if tar_path is not None:
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Target path does not exist: {tar_path}")
        tardataset = datasets.ImageFolder(root=tar_path, transform=transform)
        ft_size = int(0.7 * len(tardataset))
        ft_val_size = len(tardataset) - ft_size
        ft_dataset, ft_val_dataset = random_split(tardataset, [ft_size, ft_val_size], generator=generator)
        ft_loader = DataLoader(ft_dataset, batch_size=batch_size, shuffle=True)
        ft_val_loader = DataLoader(ft_val_dataset, batch_size=batch_size, shuffle=False)

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

        self.feature_extractor = nn.Sequential(*(list(self.model.children())[:-1]))
        self.feature_extractor.to(device)
        self.feature_extractor.eval()

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
    
    def _evaluate_embeddings(self) -> Tuple[float, float]:
        self.feature_extractor.eval()
        all_feats: List[np.ndarray] = []
        all_labels: List[int] = []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                feats = self.feature_extractor(images).squeeze(-1).squeeze(-1)  # [B, 2048]
                all_feats.append(feats.cpu().numpy())
                all_labels.append(labels.numpy())
        X = np.concatenate(all_feats, axis=0)
        y = np.concatenate(all_labels, axis=0)

        n_clusters = len(np.unique(y))
        preds = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
        ari = adjusted_rand_score(y, preds)
        nmi = normalized_mutual_info_score(y, preds)
        return ari, nmi

    def save_model(self, epoch, is_best=False):
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # 在文件名前加上 status（如 training, finetune）
        status_prefix = f"{self.status}_"

        # 文件名选择
        if is_best:
            filename = f"{status_prefix}best_model_{time_str}.pth"
        else:
            filename = f"{status_prefix}epoch_{epoch+1:03d}_{time_str}.pth"

        save_path = os.path.join(self.save_dir, filename)

        # 保存 checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, save_path)

        # 记录为最近 checkpoint
        with open(os.path.join(self.save_dir, "last_checkpoint.txt"), "w") as f:
            f.write(save_path)

        print(f"{'[BEST]' if is_best else ' [SAVE]'} Checkpoint saved: {save_path}")

        
    def train(self, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            ari, nmi = self._evaluate_embeddings()
            self.scheduler.step()

            print(self.status,f"[Epoch {epoch+1}] "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} |" 
                  f"ARI:{ari:.3f} NMI:{nmi:.3f}")
            if self.logger is not None:
                self.logger.log({"Train Loss":train_loss,"Train Acc":train_acc,"Val loss":val_loss,"Val acc":val_acc,"ARI": ari,
                    "NMI": nmi,})

            # 保存当前 epoch 的模型
            if (epoch + 1) % 10 == 0:
                self.save_model(epoch)

            # 如果当前验证准确率是最佳，保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(epoch, is_best=True)
                test=0
