# main_adapter_train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from trainer import build_dataloaders, Trainer
from models import build_model
import wandb
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--adapter_only', action='store_true')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--load_from', type=str, default=None, help='加载训练好的基础模型')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def freeze_backbone_except_adapter(model):
    for name, param in model.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False

def load_checkpoint(model, optimizer, scheduler, path):
    if os.path.exists(path):
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print(f"恢复模型: {path}")
    else:
        print(f" 未找到恢复文件: {path}")

def load_base_weights(model, path):
    if os.path.exists(path):
        print(f" 加载基础模型参数自: {path}")
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        print(f"加载失败，路径无效: {path}")

def main():
    args = parser.parse_args()
    data_root = args.data_root
    num_classes = args.num_classes
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    adapter_only = args.adapter_only
    run_name = args.run_name
    resume = args.resume
    load_from = args.load_from

    wandb.init(
        project="conv-adapter-finetune",
        name=str(run_name) if run_name else ("adapter_ft" if adapter_only else "full_ft"),
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "adapter_only": adapter_only,
            "model": "resnet50_conv_adapter"
        },
        settings=wandb.Settings(init_timeout=300)
    )

    train_loader, val_loader = build_dataloaders(data_root, batch_size)

    model = build_model(name='resnet50_conv_adapter', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if adapter_only:
        freeze_backbone_except_adapter(model)

    # ➕ 新增：加载基础模型参数
    if load_from:
        load_base_weights(model, load_from)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir="adapt_checkpoints",
        logger=wandb,
        status="adapter-only finetuning" if adapter_only else "full finetuning"
    )

    if resume:
        resume_path = "adapt_checkpoints/last_checkpoint.txt"
        if os.path.exists(resume_path):
            with open(resume_path, "r") as f:
                ckpt_path = f.read().strip()
                load_checkpoint(model, optimizer, scheduler, ckpt_path)

    trainer.train(epochs)

    # 基础模型保存
    if not adapter_only:
        with open("adapt_checkpoints/last_checkpoint.txt", "r") as f:
            best_path = f.read().strip()
        shutil.copy(best_path, "checkpoints/private1_base_model.pth")
        print(" 基础模型已保存为 checkpoints/private1_base_model.pth")

    wandb.finish()

if __name__ == "__main__":
    main()
